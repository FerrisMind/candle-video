//! Timestep and text condition embeddings for Wan transformer.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Module, VarBuilder, linear_b};

use super::feed_forward::gelu_approx_tanh;

/// Sinusoidal timestep projection (`Timesteps` in Diffusers).
pub fn sinusoidal_timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let device = timesteps.device();
    let dtype = DType::F32;
    let n = timesteps.dim(0)?;
    let half = dim / 2;
    let t = timesteps.flatten_all()?.to_dtype(dtype)?;
    let t = t.unsqueeze(1)?;

    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10_000f32.powf(i as f32 / half as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let freqs = t.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[freqs.cos()?, freqs.sin()?], D::Minus1)?;
    if dim % 2 == 1 {
        let pad = Tensor::zeros((n, 1), dtype, device)?;
        Tensor::cat(&[emb, pad], D::Minus1)
    } else {
        Ok(emb)
    }
}

#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_b(in_channels, time_embed_dim, true, vb.pp("linear_1"))?,
            linear_2: linear_b(time_embed_dim, time_embed_dim, true, vb.pp("linear_2"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?.silu()?;
        self.linear_2.forward(&x)
    }
}

#[derive(Debug)]
pub struct PixArtAlphaTextProjection {
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
}

impl PixArtAlphaTextProjection {
    pub fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_b(in_features, hidden_size, true, vb.pp("linear_1"))?,
            linear_2: linear_b(hidden_size, hidden_size, true, vb.pp("linear_2"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = gelu_approx_tanh(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&x)
    }
}

/// `WanTimeTextImageEmbedding` (T2V path — no image embedder).
#[derive(Debug)]
pub struct WanConditionEmbedder {
    time_freq_dim: usize,
    time_embedder: TimestepEmbedding,
    time_proj: candle_nn::Linear,
    text_embedder: PixArtAlphaTextProjection,
}

impl WanConditionEmbedder {
    pub fn new(
        dim: usize,
        time_freq_dim: usize,
        text_embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            time_freq_dim,
            time_embedder: TimestepEmbedding::new(time_freq_dim, dim, vb.pp("time_embedder"))?,
            time_proj: linear_b(dim, dim * 6, true, vb.pp("time_proj"))?,
            text_embedder: PixArtAlphaTextProjection::new(
                text_embed_dim,
                dim,
                vb.pp("text_embedder"),
            )?,
        })
    }

    /// Returns `(temb [B, dim], timestep_proj [B, 6, dim], text [B, seq, dim])`.
    pub fn forward(
        &self,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dtype = encoder_hidden_states.dtype();
        let ts = timestep.flatten_all()?;
        let ts_emb = sinusoidal_timestep_embedding(&ts, self.time_freq_dim)?.to_dtype(dtype)?;
        let temb = self.time_embedder.forward(&ts_emb)?;
        let timestep_proj = self.time_proj.forward(&temb.silu()?)?;
        let (b, _) = timestep_proj.dims2()?;
        let dim = temb.dim(1)?;
        let timestep_proj = timestep_proj.reshape((b, 6, dim))?;
        let text = self.text_embedder.forward(encoder_hidden_states)?;
        Ok((temb, timestep_proj, text))
    }
}
