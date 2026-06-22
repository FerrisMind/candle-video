//! Wan 3D rotary positional embeddings (`WanRotaryPosEmbed`).

use std::cell::RefCell;

use candle_core::{D, DType, Result, Tensor};

use crate::engine::wan_inference_compute_dtype;
use crate::models::wan::configs::WanTransformerConfig;

/// Cached RoPE tables for one latent grid (cos/sin + stride-selected halves for apply).
#[derive(Debug, Clone)]
pub struct WanRotaryEmb {
    pub cos: Tensor,
    pub sin: Tensor,
    cos_even: Tensor,
    sin_odd: Tensor,
}

impl WanRotaryEmb {
    pub fn as_pair(&self) -> (&Tensor, &Tensor) {
        (&self.cos, &self.sin)
    }

    fn from_cos_sin(cos: Tensor, sin: Tensor) -> Result<Self> {
        let cos_even = stride_select(&cos, 0)?;
        let sin_odd = stride_select(&sin, 1)?;
        Ok(Self {
            cos,
            sin,
            cos_even,
            sin_odd,
        })
    }
}

/// Precomputed 1D rotary frequencies for t/h/w axes.
#[derive(Debug)]
pub struct WanRotaryPosEmbed {
    freqs_cos: Tensor,
    freqs_sin: Tensor,
    patch_size: [usize; 3],
    t_dim: usize,
    h_dim: usize,
    w_dim: usize,
    head_dim: usize,
    /// ponytail: RoPE grid is resolution-stable across denoise steps — cache avoids 6× broadcast materialize per forward.
    cached: RefCell<Option<((usize, usize, usize), WanRotaryEmb)>>,
}

impl WanRotaryPosEmbed {
    pub fn new(cfg: &WanTransformerConfig, device: &candle_core::Device) -> Result<Self> {
        let head_dim = cfg.attention_head_dim;
        let max_seq_len = cfg.rope_max_seq_len;
        let h_dim = 2 * (head_dim / 6);
        let w_dim = 2 * (head_dim / 6);
        let t_dim = head_dim - h_dim - w_dim;
        // ponytail: F64 on CPU parity; F16 table on CUDA saves ~2× RoPE buffer vs F64.
        let freq_dtype = if device.is_cuda() {
            DType::F16
        } else {
            DType::F64
        };

        let freqs_cos = Tensor::cat(
            &[
                get_1d_rotary_freqs(t_dim, max_seq_len, 10_000.0, true, device, freq_dtype)?,
                get_1d_rotary_freqs(h_dim, max_seq_len, 10_000.0, true, device, freq_dtype)?,
                get_1d_rotary_freqs(w_dim, max_seq_len, 10_000.0, true, device, freq_dtype)?,
            ],
            1,
        )?;
        let freqs_sin = Tensor::cat(
            &[
                get_1d_rotary_freqs(t_dim, max_seq_len, 10_000.0, false, device, freq_dtype)?,
                get_1d_rotary_freqs(h_dim, max_seq_len, 10_000.0, false, device, freq_dtype)?,
                get_1d_rotary_freqs(w_dim, max_seq_len, 10_000.0, false, device, freq_dtype)?,
            ],
            1,
        )?;

        Ok(Self {
            freqs_cos,
            freqs_sin,
            patch_size: cfg.patch_size,
            t_dim,
            h_dim,
            w_dim,
            head_dim: t_dim + h_dim + w_dim,
            cached: RefCell::new(None),
        })
    }

    /// Input `[B, C, F, H, W]` → cached RoPE tables for all transformer blocks.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<WanRotaryEmb> {
        let (_b, _c, num_frames, height, width) = hidden_states.dims5()?;
        let [p_t, p_h, p_w] = self.patch_size;
        let key = (num_frames / p_t, height / p_h, width / p_w);
        if let Some((k, emb)) = self.cached.borrow().as_ref() {
            if *k == key {
                return Ok(emb.clone());
            }
        }
        let emb = self.compute_rope(key.0, key.1, key.2)?;
        *self.cached.borrow_mut() = Some((key, emb.clone()));
        Ok(emb)
    }

    fn compute_rope(&self, ppf: usize, pph: usize, ppw: usize) -> Result<WanRotaryEmb> {
        let head_dim = self.head_dim;

        let cos_t = self.freqs_cos.narrow(1, 0, self.t_dim)?.narrow(0, 0, ppf)?;
        let cos_h = self
            .freqs_cos
            .narrow(1, self.t_dim, self.h_dim)?
            .narrow(0, 0, pph)?;
        let cos_w = self
            .freqs_cos
            .narrow(1, self.t_dim + self.h_dim, self.w_dim)?
            .narrow(0, 0, ppw)?;

        let sin_t = self.freqs_sin.narrow(1, 0, self.t_dim)?.narrow(0, 0, ppf)?;
        let sin_h = self
            .freqs_sin
            .narrow(1, self.t_dim, self.h_dim)?
            .narrow(0, 0, pph)?;
        let sin_w = self
            .freqs_sin
            .narrow(1, self.t_dim + self.h_dim, self.w_dim)?
            .narrow(0, 0, ppw)?;

        let expand_t = |freqs: Tensor| -> Result<Tensor> {
            let (len, d) = freqs.dims2()?;
            Ok(freqs
                .reshape((len, 1, 1, d))?
                .broadcast_as((ppf, pph, ppw, d))?
                .reshape((ppf * pph * ppw, d))?)
        };
        let expand_h = |freqs: Tensor| -> Result<Tensor> {
            let (_len, d) = freqs.dims2()?;
            Ok(freqs
                .reshape((1, pph, 1, d))?
                .broadcast_as((ppf, pph, ppw, d))?
                .reshape((ppf * pph * ppw, d))?)
        };
        let expand_w = |freqs: Tensor| -> Result<Tensor> {
            let (_len, d) = freqs.dims2()?;
            Ok(freqs
                .reshape((1, 1, ppw, d))?
                .broadcast_as((ppf, pph, ppw, d))?
                .reshape((ppf * pph * ppw, d))?)
        };

        let cos_f = expand_t(cos_t)?;
        let cos_h = expand_h(cos_h)?;
        let cos_w = expand_w(cos_w)?;
        let sin_f = expand_t(sin_t)?;
        let sin_h = expand_h(sin_h)?;
        let sin_w = expand_w(sin_w)?;

        let freqs_cos = Tensor::cat(&[cos_f, cos_h, cos_w], D::Minus1)?.reshape((
            1,
            ppf * pph * ppw,
            1,
            head_dim,
        ))?;
        let freqs_sin = Tensor::cat(&[sin_f, sin_h, sin_w], D::Minus1)?.reshape((
            1,
            ppf * pph * ppw,
            1,
            head_dim,
        ))?;

        WanRotaryEmb::from_cos_sin(freqs_cos, freqs_sin)
    }
}

fn get_1d_rotary_freqs(
    dim: usize,
    max_seq_len: usize,
    theta: f64,
    cos: bool,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    assert_eq!(dim % 2, 0);
    let half = dim / 2;
    let mut exponents = Vec::with_capacity(half);
    for i in 0..half {
        exponents.push(i as f64 * 2.0 / dim as f64);
    }
    let inv_freq: Vec<f64> = exponents.iter().map(|e| 1.0 / theta.powf(*e)).collect();

    let mut out = Vec::with_capacity(max_seq_len * dim);
    for pos in 0..max_seq_len {
        for &freq in &inv_freq {
            let angle = pos as f64 * freq;
            let v = if cos { angle.cos() } else { angle.sin() };
            out.push(v as f32);
            out.push(v as f32);
        }
    }
    Tensor::from_vec(out, (max_seq_len, dim), device)?.to_dtype(dtype)
}

/// Wan-specific RoPE application (Diffusers `WanAttnProcessor`).
pub fn apply_wan_rotary_emb(hidden_states: &Tensor, rope: &WanRotaryEmb) -> Result<Tensor> {
    let dtype = hidden_states.dtype();
    let compute = wan_inference_compute_dtype(hidden_states.device(), dtype);
    let hs = hidden_states.to_dtype(compute)?;
    let (b, s, h, d) = hs.dims4()?;
    if d % 2 != 0 {
        candle_core::bail!("apply_wan_rotary_emb expects even head_dim, got {d}");
    }
    let pairs = hs.reshape((b, s, h, d / 2, 2))?;
    let x1 = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let x2 = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

    let cos_even = rope.cos_even.to_dtype(compute)?;
    let sin_odd = rope.sin_odd.to_dtype(compute)?;

    let out_even = (x1.broadcast_mul(&cos_even)? - x2.broadcast_mul(&sin_odd)?)?;
    let out_odd = (x1.broadcast_mul(&sin_odd)? + x2.broadcast_mul(&cos_even)?)?;
    Tensor::stack(&[out_even, out_odd], D::Minus1)?
        .reshape((b, s, h, d))?
        .to_dtype(dtype)
}

fn stride_select(xs: &Tensor, start: usize) -> Result<Tensor> {
    let last = xs.dim(D::Minus1)?;
    let idx: Vec<u32> = (start..last).step_by(2).map(|i| i as u32).collect();
    let device = xs.device();
    let index = Tensor::new(idx.as_slice(), device)?;
    xs.index_select(&index, D::Minus1)
}
