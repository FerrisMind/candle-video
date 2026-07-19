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
        let norm_dim = if self.channel_first {
            1
        } else {
            x.dims().len() - 1
        };
        let x32 = x.to_dtype(DType::F32)?;
        // Diffusers: F.normalize(x, dim=1) — L2 norm over channels, not RMS mean.
        let sq = x32.sqr()?;
        let sum_sq = sq.sum_keepdim(norm_dim)?;
        let l2 = sum_sq.sqrt()?.clamp(1e-12, f32::MAX as f64)?;
        let normalized = x32.broadcast_div(&l2)?;
        let mut out = normalized.affine(self.scale, 0.0)?;

        let channels = self.gamma.dim(0)?;
        let gamma = match self.gamma.rank() {
            1 if self.images => self.gamma.reshape((1, channels, 1, 1))?,
            1 => self.gamma.reshape((1, channels, 1, 1, 1))?,
            r if r >= 2 => self.gamma.unsqueeze(0)?,
            _ => candle_core::bail!("invalid gamma rank {}", self.gamma.rank()),
        };
        let gamma = gamma.to_dtype(DType::F32)?;
        out = out.broadcast_mul(&gamma)?;
        out.to_dtype(x.dtype())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Device;

    use super::*;

    #[test]
    fn zero_channel_vector_stays_finite() {
        let device = Device::Cpu;
        let weights = HashMap::from([(
            "gamma".to_string(),
            Tensor::ones((2, 1, 1, 1), DType::F32, &device).expect("gamma"),
        )]);
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let norm = WanRmsNorm::new(2, true, false, vb).expect("norm");
        let input = Tensor::zeros((1, 2, 1, 1, 1), DType::F32, &device).expect("input");

        let values = norm
            .forward(&input)
            .expect("forward")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("values");

        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| *value == 0.0));
    }
}
