//! RMSNorm across attention heads (`rms_norm_across_heads`).

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

#[derive(Debug)]
pub struct AttnRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl AttnRmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let dim = xs_f32.dim(D::Minus1)? as f64;
        let ms = xs_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let ys = xs_f32.broadcast_div(&denom)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let rank = ys.rank();
        let mut shape = vec![1usize; rank];
        shape[rank - 1] = w.dims1()?;
        ys.broadcast_mul(&w.reshape(shape)?)?.to_dtype(dtype)
    }
}

impl Module for AttnRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
