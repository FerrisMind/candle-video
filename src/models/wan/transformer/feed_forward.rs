//! Wan transformer FFN (`FeedForward` with `gelu-approximate`).

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder, linear_b};

use crate::engine::wan_inference_compute_dtype;

#[derive(Debug)]
pub struct WanFeedForward {
    proj: candle_nn::Linear,
    out: candle_nn::Linear,
}

impl WanFeedForward {
    pub fn new(dim: usize, inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: linear_b(dim, inner_dim, true, vb.pp("net").pp("0").pp("proj"))?,
            out: linear_b(inner_dim, dim, true, vb.pp("net").pp("2"))?,
        })
    }
}

pub(super) fn gelu_approx_tanh(xs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let compute = wan_inference_compute_dtype(xs.device(), dtype);
    let x = xs.to_dtype(compute)?;
    let x3 = x.powf(3.0)?;
    let inner = x.broadcast_add(&x3.affine(0.044715, 0.0)?)?;
    let inner = inner.affine((2.0 / std::f64::consts::PI).sqrt(), 0.0)?;
    let t = inner.tanh()?;
    let out = (x.affine(0.5, 0.0)? * t.affine(1.0, 1.0)?)?;
    out.to_dtype(dtype)
}

impl Module for WanFeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = gelu_approx_tanh(&self.proj.forward(xs)?)?;
        self.out.forward(&h)
    }
}
