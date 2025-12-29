//! SVD VAE Module - AutoencoderKLTemporalDecoder
//!
//! Uses standard VAE encoder and temporal-aware decoder for video.

pub mod decoder;
pub mod encoder;

// Re-exports
pub use decoder::TemporalDecoder;
pub use encoder::Encoder;

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::svd::config::SvdVaeConfig;

// Re-export DiagonalGaussianDistribution from candle-transformers
pub use candle_transformers::models::stable_diffusion::vae::DiagonalGaussianDistribution as GaussianDistribution;

/// AutoencoderKL with Temporal Decoder for SVD
///
/// This VAE uses a standard 2D encoder and a temporal-aware decoder
/// that produces temporally consistent video frames.
#[derive(Debug)]
pub struct AutoencoderKLTemporalDecoder {
    /// Standard VAE encoder
    encoder: Encoder,
    /// Quant conv for encoding
    quant_conv: candle_nn::Conv2d,
    /// Temporal decoder for video frames
    temporal_decoder: TemporalDecoder,
    config: SvdVaeConfig,
}

impl AutoencoderKLTemporalDecoder {
    pub fn new(vb: VarBuilder, config: &SvdVaeConfig) -> Result<Self> {
        // Load encoder separately (standard structure)
        let encoder = Encoder::new(vb.pp("encoder"), config)?;

        // Quant conv
        let quant_conv = candle_nn::conv2d(
            2 * config.latent_channels,
            2 * config.latent_channels,
            1,
            Default::default(),
            vb.pp("quant_conv"),
        )?;

        // Temporal decoder for video
        let temporal_decoder = TemporalDecoder::new(vb.pp("decoder"), config)?;

        Ok(Self {
            encoder,
            quant_conv,
            temporal_decoder,
            config: config.clone(),
        })
    }

    /// Encode image to latent distribution
    /// Input: [B, 3, H, W]
    /// Output: DiagonalGaussianDistribution
    pub fn encode(&self, x: &Tensor) -> Result<GaussianDistribution> {
        let x = if self.config.force_upcast && x.dtype() == DType::F16 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };

        let h = self.encoder.forward(&x)?;
        let h = self.quant_conv.forward(&h)?;
        GaussianDistribution::new(&h)
    }

    /// Encode image to latent (mode of distribution, scaled)
    /// Input: [B, 3, H, W]
    /// Output: [B, 4, H/8, W/8] scaled latents
    pub fn encode_to_latent(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();

        let x = if self.config.force_upcast && original_dtype == DType::F16 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };

        let posterior = self.encode(&x)?;
        let z = posterior.sample()?;
        let z = (z * self.config.scaling_factor)?;

        if self.config.force_upcast && original_dtype == DType::F16 {
            z.to_dtype(DType::F16)
        } else {
            Ok(z)
        }
    }

    /// Decode latents to video frames using temporal decoder
    /// Input: [B*F, 4, H/8, W/8] latents
    /// Output: [B*F, 3, H, W] frames
    /// 
    /// If chunk_size is Some, decodes frames in chunks to reduce VRAM usage.
    pub fn decode(&self, z: &Tensor, num_frames: usize, chunk_size: Option<usize>) -> Result<Tensor> {
        let original_dtype = z.dtype();
        let batch_frames = z.dim(0)?;
        let _batch_size = batch_frames / num_frames;
        let chunk_size = chunk_size.unwrap_or(num_frames);

        let z = if self.config.force_upcast && original_dtype == DType::F16 {
            z.to_dtype(DType::F32)?
        } else {
            z.clone()
        };

        let z = (z / self.config.scaling_factor)?;

        // Decode in chunks to prevent OOM (like diffusers: iterate over batch_frames dim)
        let mut decoded_chunks = Vec::new();
        for start in (0..batch_frames).step_by(chunk_size) {
            let end = std::cmp::min(start + chunk_size, batch_frames);
            let chunk_len = end - start;
            
            // Extract chunk: [chunk_len, C, H, W]
            let z_chunk = z.narrow(0, start, chunk_len)?;
            
            // For image_only_indicator, we need batch_size and num_frames_in_chunk
            // Each chunk may not align perfectly with num_frames, so we compute dynamically
            let num_frames_in_chunk = chunk_len.min(num_frames);
            let batch_for_chunk = chunk_len.div_ceil(num_frames_in_chunk);
            let image_only_indicator = Tensor::zeros((batch_for_chunk, num_frames_in_chunk), z.dtype(), z.device())?;
            
            let decoded = self.temporal_decoder.forward(&z_chunk, &image_only_indicator, num_frames_in_chunk)?;
            decoded_chunks.push(decoded);
        }

        // Concatenate all chunks
        let decoded = if decoded_chunks.len() == 1 {
            decoded_chunks.remove(0)
        } else {
            Tensor::cat(&decoded_chunks, 0)?
        };

        if self.config.force_upcast && original_dtype == DType::F16 {
            decoded.to_dtype(DType::F16)
        } else {
            Ok(decoded)
        }
    }

    pub fn scaling_factor(&self) -> f64 {
        self.config.scaling_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_config() {
        let config = SvdVaeConfig::default();
        assert_eq!(config.latent_channels, 4);
        assert_eq!(config.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(config.scaling_factor, 0.18215);
        assert!(config.force_upcast);
    }
}
