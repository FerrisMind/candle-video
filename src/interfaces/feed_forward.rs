//! Feed-forward networks for transformer models.
//!
//! Provides configurable FeedForward layers with different activation functions.

use crate::interfaces::activations::GeluProjection;
use candle_core::{Result, Tensor};
use candle_nn::{self as nn, Linear, Module, VarBuilder};

/// Feed-forward network with GELU approximate activation.
///
/// LTX-style FeedForward: Linear+GELU_approx -> Linear
#[derive(Clone, Debug)]
pub struct FeedForward {
    net_0: GeluProjection,
    net_2: Linear,
}

impl FeedForward {
    pub fn new_gelu(dim: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * mult;
        let net_0 = GeluProjection::new(dim, hidden, vb.pp("net.0"))?;
        let net_2 = nn::linear(hidden, dim, vb.pp("net.2"))?;
        Ok(Self { net_0, net_2 })
    }

    /// Create FeedForward with default configuration (mult=4).
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_gelu(dim, 4, vb)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.net_0.forward(xs)?;
        self.net_2.forward(&x)
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_feed_forward_shapes() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let ff = FeedForward::new(64, vb.pp("ff"))?;
        let x = Tensor::zeros((2, 10, 64), DType::F32, &device)?;
        let out = ff.forward(&x)?;
        assert_eq!(out.dims(), &[2, 10, 64]);
        Ok(())
    }
}
