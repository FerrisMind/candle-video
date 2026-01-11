//! Embedding layers for transformer models.
//!
//! Provides timestep embeddings, text projections, and adaptive normalization
//! layers used across video generation models.

use crate::interfaces::activations::gelu_approximate;
use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{self as nn, Linear, VarBuilder};

/// Sinusoidal timestep embeddings (DDPM-style).
///
/// Creates sinusoidal position embeddings for diffusion timesteps.
pub fn get_timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
) -> Result<Tensor> {
    let device = timesteps.device();
    let original_dtype = timesteps.dtype();
    let dtype = DType::F32;

    let n = timesteps.dim(0)?;
    let half = embedding_dim / 2;

    let t = timesteps.to_dtype(dtype)?;
    let t = t.unsqueeze(1)?;

    let inv_freq: Vec<_> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / (half as f32)))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
    let freqs = t.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let sin = freqs.sin()?;
    let cos = freqs.cos()?;

    let emb = if flip_sin_to_cos {
        Tensor::cat(&[cos, sin], D::Minus1)?
    } else {
        Tensor::cat(&[sin, cos], D::Minus1)?
    };

    if embedding_dim % 2 == 1 {
        let pad = Tensor::zeros((n, 1), dtype, device)?;
        Tensor::cat(&[emb, pad], D::Minus1)?.to_dtype(original_dtype)
    } else {
        emb.to_dtype(original_dtype)
    }
}

/// Timestep embedding with two linear layers and SiLU.
#[derive(Clone, Debug)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = x.silu()?;
        self.linear_2.forward(&x)
    }
}

/// PixArt-Alpha text projection (two linear layers with GELU).
#[derive(Clone, Debug)]
pub struct PixArtAlphaTextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl PixArtAlphaTextProjection {
    pub fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_features, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = gelu_approximate(&x)?;
        self.linear_2.forward(&x)
    }
}

/// PixArt-Alpha combined timestep + size embeddings.
#[derive(Clone, Debug)]
pub struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    timestep_embedder: TimestepEmbedding,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder =
            TimestepEmbedding::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self { timestep_embedder })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let timesteps_proj = get_timestep_embedding(timestep, 256, true)?;
        self.timestep_embedder.forward(&timesteps_proj)
    }
}

/// Adaptive Layer Normalization (Single) for DiT architectures.
///
/// Returns 6 scale/shift/gate parameters for attention and FFN.
#[derive(Clone, Debug)]
pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: Linear,
}

impl AdaLayerNormSingle {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(dim, vb.pp("emb"))?;
        let linear = nn::linear(dim, 6 * dim, vb.pp("linear"))?;
        Ok(Self { emb, linear })
    }

    /// Returns (ada_params, embedded_timestep) where ada_params has 6*dim values.
    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded_timestep = self.emb.forward(timestep)?;
        let x = embedded_timestep.silu()?;
        let x = self.linear.forward(&x)?;
        Ok((x, embedded_timestep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_get_timestep_embedding() -> Result<()> {
        let device = Device::Cpu;
        let timesteps = Tensor::new(&[0.0f32, 0.5, 1.0], &device)?;
        let emb = get_timestep_embedding(&timesteps, 64, true)?;
        assert_eq!(emb.dims(), &[3, 64]);
        Ok(())
    }

    #[test]
    fn test_get_timestep_embedding_odd_dim() -> Result<()> {
        let device = Device::Cpu;
        let timesteps = Tensor::new(&[0.5f32], &device)?;
        let emb = get_timestep_embedding(&timesteps, 65, false)?;
        assert_eq!(emb.dims(), &[1, 65]);
        Ok(())
    }
}
