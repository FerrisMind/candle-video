//! Wan attention (`WanAttention` + `WanAttnProcessor`).

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Dropout, Module, VarBuilder, linear_b};

use super::attn_rms_norm::AttnRmsNorm;
use super::rope::{WanRotaryEmb, apply_wan_rotary_emb};

#[derive(Debug)]
pub struct WanAttention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    _kv_inner_dim: usize,
    norm_q: AttnRmsNorm,
    norm_k: AttnRmsNorm,
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    dropout: Dropout,
    is_cross_attention: bool,
}

impl WanAttention {
    pub fn new_self_attn(
        dim: usize,
        heads: usize,
        head_dim: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = head_dim * heads;
        Ok(Self {
            heads,
            head_dim,
            inner_dim,
            _kv_inner_dim: inner_dim,
            norm_q: AttnRmsNorm::new(inner_dim, eps, vb.pp("norm_q"))?,
            norm_k: AttnRmsNorm::new(inner_dim, eps, vb.pp("norm_k"))?,
            to_q: linear_b(dim, inner_dim, true, vb.pp("to_q"))?,
            to_k: linear_b(dim, inner_dim, true, vb.pp("to_k"))?,
            to_v: linear_b(dim, inner_dim, true, vb.pp("to_v"))?,
            to_out: linear_b(inner_dim, dim, true, vb.pp("to_out").pp("0"))?,
            dropout: Dropout::new(0.0),
            is_cross_attention: false,
        })
    }

    pub fn new_cross_attn(
        dim: usize,
        heads: usize,
        head_dim: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = head_dim * heads;
        let cross_head_dim = head_dim;
        let kv_inner_dim = cross_head_dim * heads;
        Ok(Self {
            heads,
            head_dim,
            inner_dim,
            _kv_inner_dim: kv_inner_dim,
            norm_q: AttnRmsNorm::new(inner_dim, eps, vb.pp("norm_q"))?,
            norm_k: AttnRmsNorm::new(inner_dim, eps, vb.pp("norm_k"))?,
            to_q: linear_b(dim, inner_dim, true, vb.pp("to_q"))?,
            to_k: linear_b(dim, kv_inner_dim, true, vb.pp("to_k"))?,
            to_v: linear_b(dim, kv_inner_dim, true, vb.pp("to_v"))?,
            to_out: linear_b(inner_dim, dim, true, vb.pp("to_out").pp("0"))?,
            dropout: Dropout::new(0.0),
            is_cross_attention: true,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        rotary_emb: Option<&WanRotaryEmb>,
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let enc = encoder_hidden_states.unwrap_or(hidden_states);
        let (_, k_len, _) = enc.dims3()?;

        let mut q = self.to_q.forward(hidden_states)?;
        let mut k = self.to_k.forward(enc)?;
        let v = self.to_v.forward(enc)?;

        q = self.norm_q.forward(&q)?;
        k = self.norm_k.forward(&k)?;

        let q = q.reshape((b, q_len, self.heads, self.head_dim))?;
        let k = k.reshape((b, k_len, self.heads, self.head_dim))?;
        let v = v.reshape((b, k_len, self.heads, self.head_dim))?;

        let (q, k) = if let Some(rope) = rotary_emb {
            if self.is_cross_attention {
                (q, k)
            } else {
                (
                    apply_wan_rotary_emb(&q, rope)?,
                    apply_wan_rotary_emb(&k, rope)?,
                )
            }
        } else {
            (q, k)
        };

        let dtype = q.dtype();
        let scale = 1f32 / (self.head_dim as f32).sqrt();

        crate::engine::require_flash_attn_for_cuda(q.device(), "WanAttention", q_len.max(k_len))?;

        #[allow(unused_mut)]
        let mut use_flash = false;
        #[cfg(feature = "flash-attn")]
        {
            if q.device().is_cuda() {
                use_flash = true;
            }
        }

        let out = if use_flash {
            #[cfg(feature = "flash-attn")]
            {
                // [B, seq, heads, head_dim] — avoids materializing O(seq²) attention on GPU.
                let flash_dtype = if dtype == DType::F32 {
                    DType::F16
                } else {
                    dtype
                };
                // ponytail: q/k/v come from a Linear+reshape, so they are
                // already contiguous in their native dtype. Avoid the
                // contiguous() + to_dtype() rebuilds on the hot path.
                let q_owned = if q.is_contiguous() && q.dtype() == flash_dtype {
                    None
                } else {
                    Some(q.contiguous()?.to_dtype(flash_dtype)?)
                };
                let q_fa: &Tensor = q_owned.as_ref().unwrap_or(q);
                let k_owned = if k.is_contiguous() && k.dtype() == flash_dtype {
                    None
                } else {
                    Some(k.contiguous()?.to_dtype(flash_dtype)?)
                };
                let k_fa: &Tensor = k_owned.as_ref().unwrap_or(k);
                let v_owned = if v.is_contiguous() && v.dtype() == flash_dtype {
                    None
                } else {
                    Some(v.contiguous()?.to_dtype(flash_dtype)?)
                };
                let v_fa: &Tensor = v_owned.as_ref().unwrap_or(v);
                let out = candle_flash_attn::flash_attn(q_fa, k_fa, v_fa, scale, false)?;
                drop((q_owned, k_owned, v_owned));
                out.reshape((b, q_len, self.inner_dim))?.to_dtype(dtype)?
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                unreachable!()
            }
        } else {
            // CPU parity path uses F32; CUDA without flash-attn uses F16 to avoid O(seq²) F32 peaks.
            let matmul_dtype = if q.device().is_cuda() {
                DType::F16
            } else {
                DType::F32
            };
            let q_m = q.transpose(1, 2)?.contiguous()?.to_dtype(matmul_dtype)?;
            let k_m = k.transpose(1, 2)?.contiguous()?.to_dtype(matmul_dtype)?;
            let v_m = v.transpose(1, 2)?.contiguous()?.to_dtype(matmul_dtype)?;

            let k_t = k_m.transpose(D::Minus1, D::Minus2)?.contiguous()?;
            let att = q_m.matmul(&k_t)?;
            let att = (att * (scale as f64))?;
            let (b_sz, h_sz, q_l, k_l) = att.dims4()?;
            let att = att.reshape((b_sz * h_sz * q_l, k_l))?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            let att = att.reshape((b_sz, h_sz, q_l, k_l))?;
            let out_m = att.matmul(&v_m.contiguous()?)?;

            out_m
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, q_len, self.inner_dim))?
                .to_dtype(dtype)?
        };

        let out = self.to_out.forward(&out)?;
        self.dropout.forward(&out, false)
    }
}
