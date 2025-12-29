//! CLIP Vision Model with Projection for SVD
//!
//! Wraps candle-transformers ClipVisionTransformer and adds a projection layer
//! to produce embeddings of dimension `projection_dim` (1024 for SVD).

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::clip::text_model::Activation;
use candle_transformers::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};

use crate::svd::config::ClipEncoderConfig;

/// CLIP Vision Model with Projection head for SVD conditioning
///
/// Takes input images and produces embeddings of shape [B, projection_dim].
/// The projection head projects from hidden_size (1280) to projection_dim (1024).
#[derive(Debug)]
pub struct ClipVisionModelWithProjection {
    vision_model: ClipVisionTransformer,
    visual_projection: Linear,
    config: ClipEncoderConfig,
}

impl ClipVisionModelWithProjection {
    pub fn new(vb: VarBuilder, config: &ClipEncoderConfig) -> Result<Self> {
        // Convert our config to ClipVisionConfig for candle-transformers
        // Note: candle-transformers CLIP only supports QuickGelu activation
        let clip_config = ClipVisionConfig {
            embed_dim: config.hidden_size,
            activation: Activation::QuickGelu,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            projection_dim: config.projection_dim,
            num_channels: config.num_channels,
            image_size: config.image_size,
            patch_size: config.patch_size,
        };

        let vision_model = ClipVisionTransformer::new(vb.pp("vision_model"), &clip_config)?;

        // Projection layer: hidden_size -> projection_dim (no bias in SVD)
        let visual_projection = candle_nn::linear_no_bias(
            config.hidden_size,
            config.projection_dim,
            vb.pp("visual_projection"),
        )?;

        Ok(Self {
            vision_model,
            visual_projection,
            config: config.clone(),
        })
    }

    /// Forward pass: pixel_values [B, 3, 224, 224] -> embeddings [B, projection_dim]
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Vision transformer produces pooled output [B, hidden_size]
        let pooled_output = self.vision_model.forward(pixel_values)?;

        // Project to embedding dimension [B, projection_dim]
        let image_embeds = self.visual_projection.forward(&pooled_output)?;

        Ok(image_embeds)
    }

    /// Get hidden states from all layers (useful for some applications)
    pub fn forward_with_hidden_states(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        self.vision_model.output_hidden_states(pixel_values)
    }

    /// Get the projection dimension
    pub fn projection_dim(&self) -> usize {
        self.config.projection_dim
    }

    /// Get the expected image size
    pub fn image_size(&self) -> usize {
        self.config.image_size
    }
}

/// CLIP image preprocessing constants
pub mod preprocessing {
    /// CLIP normalization mean (ImageNet)
    pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    /// CLIP normalization std (ImageNet)  
    pub const CLIP_STD: [f32; 3] = [0.268_629_5, 0.261_302_6, 0.275_777_1];
}

/// Normalize image tensor with CLIP means and stds
/// Input: [B, 3, H, W] in range [0, 1]
/// Output: [B, 3, H, W] normalized
pub fn normalize_for_clip(images: &Tensor, device: &Device) -> Result<Tensor> {
    let mean = Tensor::new(&preprocessing::CLIP_MEAN, device)?
        .reshape((1, 3, 1, 1))?
        .to_dtype(images.dtype())?;
    let std = Tensor::new(&preprocessing::CLIP_STD, device)?
        .reshape((1, 3, 1, 1))?
        .to_dtype(images.dtype())?;

    (images.broadcast_sub(&mean))?.broadcast_div(&std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_clip_config() {
        let config = ClipEncoderConfig::default();
        assert_eq!(config.hidden_size, 1280);
        assert_eq!(config.projection_dim, 1024);
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 14);
    }

    #[test]
    fn test_normalize_for_clip() {
        let device = Device::Cpu;
        // Create a dummy image tensor [1, 3, 4, 4]
        let images = Tensor::ones((1, 3, 4, 4), DType::F32, &device).unwrap();

        let normalized = normalize_for_clip(&images, &device).unwrap();

        // Check shape is preserved
        assert_eq!(normalized.dims(), &[1, 3, 4, 4]);

        // Mean should be subtracted, so values should be different
        let orig_mean = images.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let norm_mean = normalized.mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((orig_mean - norm_mean).abs() > 0.1);
    }
}
