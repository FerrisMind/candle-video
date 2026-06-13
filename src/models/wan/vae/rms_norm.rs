//! Wan RMS normalization (`WanRMS_norm` in Diffusers).

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// RMS norm with learnable `gamma` (Diffusers key: `.gamma`).
#[derive(Debug)]
pub struct WanRmsNorm {
    gamma: Tensor,
    scale: f64,
    channel_first: bool,
    images: bool,
}

impl WanRmsNorm {
    pub fn new(dim: usize, channel_first: bool, images: bool, vb: VarBuilder) -> Result<Self> {
        let gamma = if images {
            vb.get((dim, 1, 1), "gamma")
                .or_else(|_| vb.get(dim, "gamma"))?
        } else {
            vb.get((dim, 1, 1, 1), "gamma")
                .or_else(|_| vb.get(dim, "gamma"))?
        };
        let _ = channel_first;
        Ok(Self {
            gamma,
            scale: (dim as f64).sqrt(),
            channel_first,
            images,
        })
    }
}

impl Module for WanRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_dim = if self.channel_first { 1 } else { x.dims().len() - 1 };
        let x32 = x.to_dtype(DType::F32)?;
        // Diffusers: F.normalize(x, dim=1) — L2 norm over channels, not RMS mean.
        let sq = x32.sqr()?;
        let sum_sq = sq.sum_keepdim(norm_dim)?;
        let l2 = sum_sq.sqrt()?;
        let normalized = x32.broadcast_div(&l2)?;
        let mut out = normalized.affine(self.scale, 0.0)?;

        let channels = self.gamma.dim(0)?;
        let gamma = match self.gamma.rank() {
            1 if self.images => self.gamma.reshape((1, channels, 1, 1))?,
            1 => self.gamma.reshape((1, channels, 1, 1, 1))?,
            r if r >= 2 => self.gamma.unsqueeze(0)?,
            _ => candle_core::bail!("invalid gamma rank {}", self.gamma.rank()),
        };
        out = out.broadcast_mul(&gamma)?;
        out.to_dtype(x.dtype())
    }
}
