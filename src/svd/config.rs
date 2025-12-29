//! Configuration structures for SVD components

use serde::{Deserialize, Serialize};

/// Main SVD pipeline configuration
#[derive(Debug, Clone, Default)]
pub struct SvdConfig {
    pub unet: SvdUnetConfig,
    pub vae: SvdVaeConfig,
    pub scheduler: EulerSchedulerConfig,
    pub clip: ClipEncoderConfig,
}

/// UNet spatio-temporal condition model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvdUnetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub cross_attention_dim: usize,
    pub num_attention_heads: Vec<usize>,
    pub num_frames: usize,
    pub addition_time_embed_dim: usize,
    pub projection_class_embeddings_input_dim: usize,
    pub transformer_layers_per_block: usize,
    pub sample_size: usize,
}

impl Default for SvdUnetConfig {
    fn default() -> Self {
        Self {
            in_channels: 8,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            cross_attention_dim: 1024,
            num_attention_heads: vec![5, 10, 20, 20],
            num_frames: 14,
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 768,
            transformer_layers_per_block: 1,
            sample_size: 96,
        }
    }
}

/// VAE with temporal decoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvdVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f64,
    pub force_upcast: bool,
    pub sample_size: usize,
}

impl Default for SvdVaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.18215,
            force_upcast: true,
            sample_size: 768,
        }
    }
}

/// Euler discrete scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EulerSchedulerConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: String,
    pub prediction_type: String,
    pub timestep_spacing: String,
    pub timestep_type: String,
    pub steps_offset: usize,
    pub use_karras_sigmas: bool,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub interpolation_type: String,
}

impl Default for EulerSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            prediction_type: "v_prediction".to_string(),
            timestep_spacing: "leading".to_string(),
            timestep_type: "continuous".to_string(),
            steps_offset: 1,
            use_karras_sigmas: true,
            sigma_min: 0.002,
            sigma_max: 700.0,
            interpolation_type: "linear".to_string(),
        }
    }
}

/// CLIP vision encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub projection_dim: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
}

impl Default for ClipEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 224,
            patch_size: 14,
            projection_dim: 1024,
            num_channels: 3,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
        }
    }
}

/// SVD inference parameters (runtime configuration)
#[derive(Debug, Clone)]
pub struct SvdInferenceConfig {
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub min_guidance_scale: f64,
    pub max_guidance_scale: f64,
    pub fps: usize,
    pub motion_bucket_id: usize,
    pub noise_aug_strength: f64,
    pub decode_chunk_size: Option<usize>,
    pub seed: u64,
}

impl Default for SvdInferenceConfig {
    fn default() -> Self {
        Self {
            num_frames: 14,
            height: 576,
            width: 1024,
            num_inference_steps: 25,
            min_guidance_scale: 1.0,
            max_guidance_scale: 3.0,
            fps: 7,
            motion_bucket_id: 127,
            noise_aug_strength: 0.02,
            decode_chunk_size: Some(2), // Chunk decode to prevent OOM
            seed: 42,
        }
    }
}

impl SvdInferenceConfig {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_steps(mut self, steps: usize) -> Self {
        self.num_inference_steps = steps;
        self
    }

    pub fn with_guidance(mut self, min: f64, max: f64) -> Self {
        self.min_guidance_scale = min;
        self.max_guidance_scale = max;
        self
    }

    pub fn with_motion(mut self, fps: usize, motion_bucket_id: usize) -> Self {
        self.fps = fps;
        self.motion_bucket_id = motion_bucket_id;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let config = SvdConfig::default();
        assert_eq!(config.unet.in_channels, 8);
        assert_eq!(config.unet.out_channels, 4);
        assert_eq!(config.unet.num_frames, 14);
        assert_eq!(config.vae.latent_channels, 4);
        assert_eq!(config.vae.scaling_factor, 0.18215);
        assert!(config.vae.force_upcast);
        assert_eq!(config.scheduler.prediction_type, "v_prediction");
        assert!(config.scheduler.use_karras_sigmas);
        assert_eq!(config.clip.projection_dim, 1024);
    }

    #[test]
    fn test_inference_config_builder() {
        let config = SvdInferenceConfig::default()
            .with_seed(123)
            .with_steps(30)
            .with_guidance(1.5, 4.0)
            .with_motion(8, 200);

        assert_eq!(config.seed, 123);
        assert_eq!(config.num_inference_steps, 30);
        assert_eq!(config.min_guidance_scale, 1.5);
        assert_eq!(config.max_guidance_scale, 4.0);
        assert_eq!(config.fps, 8);
        assert_eq!(config.motion_bucket_id, 200);
    }
}
