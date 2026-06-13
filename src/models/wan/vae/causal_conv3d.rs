//! Wan causal 3D convolution with zero padding and feature-cache support.

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

pub const CACHE_T: usize = 2;

/// Feature-cache slot used during frame-by-frame decode.
#[derive(Clone, Debug)]
pub enum FeatSlot {
    Empty,
    /// First pass through `upsample3d` time conv (no temporal mix yet).
    Rep,
    Tensor(Tensor),
}

/// Mutable feature cache shared across a single latent frame decode.
pub struct FeatCache {
    pub slots: Vec<FeatSlot>,
    pub idx: usize,
}

impl FeatCache {
    pub fn new(num_convs: usize) -> Self {
        Self {
            slots: vec![FeatSlot::Empty; num_convs],
            idx: 0,
        }
    }

    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = FeatSlot::Empty;
        }
        self.idx = 0;
    }

    pub fn reset_idx(&mut self) {
        self.idx = 0;
    }

    pub fn next_slot(&mut self) -> usize {
        let i = self.idx;
        self.idx += 1;
        i
    }

    pub(crate) fn get(&self, i: usize) -> &FeatSlot {
        &self.slots[i]
    }

    pub(crate) fn set(&mut self, i: usize, slot: FeatSlot) {
        self.slots[i] = slot;
    }
}

fn cat_time(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }
    Tensor::cat(tensors, 2)
}

fn pad_wan(x: &Tensor, pad: (usize, usize, usize, usize, usize, usize)) -> Result<Tensor> {
    // Input `[B,C,T,H,W]`; pad order matches PyTorch `F.pad` on last 3 dims: W, H, T.
    let (pl_w, pr_w, pl_h, pr_h, pl_t, pr_t) = pad;
    let mut out = x.clone();
    if pl_w + pr_w > 0 {
        out = out.pad_with_zeros(4, pl_w, pr_w)?;
    }
    if pl_h + pr_h > 0 {
        out = out.pad_with_zeros(3, pl_h, pr_h)?;
    }
    if pl_t + pr_t > 0 {
        out = out.pad_with_zeros(2, pl_t, pr_t)?;
    }
    Ok(out)
}

/// Diffusers `WanCausalConv3d` — zero pad (`2 * pt` on the time front), not LTX replicate.
#[derive(Debug)]
pub struct WanCausalConv3d {
    kt: usize,
    stride_t: usize,
    dil_t: usize,
    conv2d_slices: Vec<Conv2d>,
    bias: Tensor,
    padding: (usize, usize, usize),
}

impl WanCausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let (st, _sh, _sw) = stride;
        let w = vb.get((out_channels, in_channels, kt, kh, kw), "weight")?;
        let bias = vb.get(out_channels, "bias")?;

        let mut conv2d_slices = Vec::with_capacity(kt);
        for ti in 0..kt {
            let w2 = w.i((.., .., ti, .., ..))?.contiguous()?;
            let cfg = Conv2dConfig {
                padding: 0,
                stride: 1,
                ..Default::default()
            };
            conv2d_slices.push(Conv2d::new(w2, None, cfg));
        }

        Ok(Self {
            kt,
            stride_t: st,
            dil_t: 1,
            conv2d_slices,
            bias,
            padding,
        })
    }

    fn wan_padding(&self) -> (usize, usize, usize, usize, usize, usize) {
        let (pt, ph, pw) = self.padding;
        (pw, pw, ph, ph, 2 * pt, 0)
    }

    fn prepare_input(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<Tensor> {
        let (pl_w, pr_w, pl_h, pr_h, mut pl_t, pr_t) = self.wan_padding();
        let mut x = x.clone();
        if let Some(cache_x) = cache
            && pl_t > 0
        {
            x = Tensor::cat(&[cache_x, &x], 2)?;
            pl_t = pl_t.saturating_sub(cache_x.dim(2)?);
        }
        pad_wan(&x, (pl_w, pr_w, pl_h, pr_h, pl_t, pr_t))
    }

    pub fn forward(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<Tensor> {
        let x = self.prepare_input(x, cache)?;
        let (_b, _c, t_pad, _h, _w) = x.dims5()?;
        let kt = self.kt;
        let st = self.stride_t;
        let dt = self.dil_t;
        let needed = (kt - 1) * dt + 1;
        if t_pad < needed {
            candle_core::bail!(
                "time dim too small after padding: t_pad={t_pad}, needed={needed}"
            );
        }
        let t_out = (t_pad - needed) / st + 1;
        let mut ys = Vec::with_capacity(t_out);
        for to in 0..t_out {
            let base_t = to * st;
            let mut acc: Option<Tensor> = None;
            for ki in 0..kt {
                let ti = base_t + ki * dt;
                let xt = x.i((.., .., ti, .., ..))?;
                let yt = xt.apply(&self.conv2d_slices[ki])?;
                acc = Some(match acc {
                    None => yt,
                    Some(prev) => prev.add(&yt)?,
                });
            }
            ys.push(acc.expect("kt>=1").unsqueeze(2)?);
        }
        let y = cat_time(&ys)?;
        let bias = self.bias.reshape((1, self.bias.dims1()?, 1, 1, 1))?;
        y.broadcast_add(&bias)
    }

    pub fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut FeatCache,
        prepad_cache: bool,
    ) -> Result<Tensor> {
        let idx = cache.next_slot();
        let slot = cache.get(idx).clone();
        let prev = match slot {
            FeatSlot::Tensor(t) => Some(t),
            _ => None,
        };
        if prepad_cache {
            let t_in = x.dim(2)?;
            let mut cache_x = x.i((.., .., t_in.saturating_sub(CACHE_T).., .., ..))?;
            if cache_x.dim(2)? < CACHE_T
                && let Some(ref prev_t) = prev
            {
                let last = prev_t.i((.., .., prev_t.dim(2)? - 1.., .., ..))?;
                cache_x = Tensor::cat(&[&last, &cache_x], 2)?;
            }
            let out = self.forward(x, prev.as_ref())?;
            cache.set(idx, FeatSlot::Tensor(cache_x));
            return Ok(out);
        }
        self.forward(x, prev.as_ref())
    }
}

pub fn count_causal_conv3d(decoder: &super::decoder::WanDecoder3d) -> usize {
    decoder.count_causal_conv3d()
}
