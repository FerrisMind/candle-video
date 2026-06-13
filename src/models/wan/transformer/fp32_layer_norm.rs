//! FP32 LayerNorm matching Diffusers `FP32LayerNorm`.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// LayerNorm computed in F32, matching Diffusers `FP32LayerNorm`.
#[derive(Debug)]
pub struct Fp32LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
    normalized_dim: usize,
}

impl Fp32LayerNorm {
    pub fn new(dim: usize, eps: f64, elementwise_affine: bool, vb: VarBuilder) -> Result<Self> {
        let (weight, bias) = if elementwise_affine {
            (
                Some(vb.get(dim, "weight")?),
                Some(vb.get(dim, "bias")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            weight,
            bias,
            eps,
            normalized_dim: dim,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let mean = xs_f32.mean_keepdim(D::Minus1)?;
        let xc = xs_f32.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let mut ys = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;

        if let Some(w) = &self.weight {
            let w = w.to_dtype(DType::F32)?;
            ys = ys.broadcast_mul(&w.broadcast_as(ys.shape())?)?;
        }
        if let Some(b) = &self.bias {
            let b = b.to_dtype(DType::F32)?;
            ys = ys.broadcast_add(&b.broadcast_as(ys.shape())?)?;
        }
        let _ = self.normalized_dim;
        ys.to_dtype(dtype)
    }
}

impl Module for Fp32LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
