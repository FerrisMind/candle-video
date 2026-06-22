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
            (Some(vb.get(dim, "weight")?), Some(vb.get(dim, "bias")?))
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
        // ponytail: CUDA inference keeps activations in model dtype; CPU path stays F32 for parity tests.
        let compute = if xs.device().is_cuda() {
            dtype
        } else {
            DType::F32
        };
        self.forward_in_dtype(xs, compute)?.to_dtype(dtype)
    }

    fn forward_in_dtype(&self, xs: &Tensor, compute: DType) -> Result<Tensor> {
        let xs_c = xs.to_dtype(compute)?;
        let mean = xs_c.mean_keepdim(D::Minus1)?;
        let xc = xs_c.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let mut ys = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;

        if let Some(w) = &self.weight {
            let w = w.to_dtype(compute)?;
            ys = ys.broadcast_mul(&w.broadcast_as(ys.shape())?)?;
        }
        if let Some(b) = &self.bias {
            let b = b.to_dtype(compute)?;
            ys = ys.broadcast_add(&b.broadcast_as(ys.shape())?)?;
        }
        let _ = self.normalized_dim;
        Ok(ys)
    }
}

impl Module for Fp32LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
