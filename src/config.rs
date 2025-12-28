//! Configuration structures for LTX-Video components

use serde::{Deserialize, Serialize};

/// VAE configuration for causal video VAE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaeConfig {
    /// Input channels (usually 3 for RGB)
    pub in_channels: usize,
    /// Output channels (usually 3 for RGB)
    pub out_channels: usize,
    /// Latent space channels (usually 128 for LTX)
    pub latent_channels: usize,
    /// Output channels for each block
    pub block_out_channels: Vec<usize>,
    /// Number of layers per block
    pub layers_per_block: usize,
    /// Temporal downsampling factor (usually 8)
    pub temporal_downsample: usize,
    /// Spatial downsampling factor (usually 32)
    pub spatial_downsample: usize,
    /// Whether to use causal padding
    pub causal: bool,
    /// Latents mean for normalization (per channel, 128 values for LTX)
    #[serde(default)]
    pub latents_mean: Option<Vec<f64>>,
    /// Latents std for normalization (per channel, 128 values for LTX)
    #[serde(default)]
    pub latents_std: Option<Vec<f64>>,
    /// Scaling factor for latent normalization (usually 1.0)
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f64,
    /// Whether to use timestep conditioning in VAE decoder
    #[serde(default)]
    pub timestep_conditioning: bool,
}

fn default_scaling_factor() -> f64 {
    1.0
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            temporal_downsample: 8,
            spatial_downsample: 32,
            causal: true,
            latents_mean: None,
            latents_std: None,
            scaling_factor: 1.0,
            timestep_conditioning: false,
        }
    }
}

/// DiT (Diffusion Transformer) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DitConfig {
    /// Spatial patch size for patch embedding (applies to H, W)
    pub patch_size: usize,
    /// Temporal patch size for patch embedding (applies to T)
    /// If None, uses patch_size for all dimensions
    #[serde(default)]
    pub patch_size_t: Option<usize>,
    /// Input channels (latent channels from VAE)
    pub in_channels: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of transformer layers
    pub depth: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Caption channels (T5-XXL output size, usually 4096)
    pub caption_channels: usize,
    /// MLP ratio for feedforward network
    pub mlp_ratio: f64,
    /// Whether to use flash attention
    pub use_flash_attention: bool,
    /// Timestep scale multiplier (1000.0 for LTX-Video 2B v0.9.8)
    /// Applied before AdaLayerNormSingle: timestep = timestep * multiplier
    #[serde(default)]
    pub timestep_scale_multiplier: Option<f64>,
}

impl Default for DitConfig {
    fn default() -> Self {
        Self {
            patch_size: 2,
            patch_size_t: None, // Uses patch_size by default
            in_channels: 128,
            hidden_size: 1152,
            depth: 28,
            num_heads: 16,
            caption_channels: 4096,
            mlp_ratio: 4.0,
            use_flash_attention: true,
            timestep_scale_multiplier: Some(1000.0),
        }
    }
}

/// Rectified Flow scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Number of training timesteps (typically 1000)
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    /// Guidance scale for classifier-free guidance
    pub guidance_scale: f64,
    /// Timestep spacing type: "linspace" or "linear_quadratic"
    pub timestep_spacing: String,
    /// Sigma schedule strategy: "linspace" (default), "karras", "exponential", "beta"
    /// - karras: Better quality at low step counts (8-10 steps)
    /// - exponential: More fine-grained control at low noise levels
    /// - beta: Flexible distribution via alpha/beta parameters
    #[serde(default)]
    pub sigma_schedule: Option<String>,
    /// Shift value for timestep schedule (None = no shift, 1.0 = identity)
    #[serde(default)]
    pub shift: Option<f64>,
    /// Whether to apply dynamic resolution-dependent timestep shifting
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    /// Base shift for dynamic shifting
    #[serde(default = "default_base_shift")]
    pub base_shift: f64,
    /// Max shift for dynamic shifting
    #[serde(default = "default_max_shift")]
    pub max_shift: f64,
    /// Terminal value for shifted timestep schedule
    #[serde(default)]
    pub shift_terminal: Option<f64>,
    /// Whether to use stochastic sampling (adds noise during sampling)
    #[serde(default)]
    pub stochastic_sampling: bool,
}

fn default_num_train_timesteps() -> usize {
    1000
}

fn default_base_shift() -> f64 {
    0.95 // ComfyUI default for LTX-Video
}

fn default_max_shift() -> f64 {
    2.05 // ComfyUI default for LTX-Video
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            num_train_timesteps: 1000,
            guidance_scale: 3.0,
            timestep_spacing: "linspace".to_string(),
            sigma_schedule: None, // None = use timestep_spacing directly
            shift: None,
            use_dynamic_shifting: true, // Enable dynamic shifting by default
            base_shift: 0.95,
            max_shift: 2.05,
            shift_terminal: Some(0.10), // ComfyUI default terminal
            stochastic_sampling: false,
        }
    }
}

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Number of frames to generate (must be 8N+1)
    pub num_frames: usize,
    /// Frame rate for RoPE scaling (None = default)
    pub frame_rate: Option<f64>,
    /// Video height (must be multiple of 32)
    pub height: usize,
    /// Video width (must be multiple of 32)
    pub width: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Guidance scale
    pub guidance_scale: f64,
}

impl InferenceConfig {
    /// Create a new inference configuration with validation
    pub fn new(
        num_frames: usize,
        height: usize,
        width: usize,
        seed: u64,
    ) -> Result<Self, ConfigError> {
        // Validate frame count (must be 8N+1)
        #[allow(clippy::manual_is_multiple_of)]
        if (num_frames - 1) % 8 != 0 {
            return Err(ConfigError::InvalidFrameCount(num_frames));
        }

        // Validate dimensions (must be multiple of 32)
        #[allow(clippy::manual_is_multiple_of)]
        if height % 32 != 0 || width % 32 != 0 {
            return Err(ConfigError::InvalidDimensions { height, width });
        }

        Ok(Self {
            num_frames,
            frame_rate: None,
            height,
            width,
            seed,
            num_inference_steps: 50,
            guidance_scale: 3.0,
        })
    }

    /// Set number of inference steps
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.num_inference_steps = steps;
        self
    }

    /// Set guidance scale
    pub fn with_guidance_scale(mut self, scale: f64) -> Self {
        self.guidance_scale = scale;
        self
    }

    /// Set frame rate for RoPE scaling
    pub fn with_frame_rate(mut self, frame_rate: f64) -> Self {
        self.frame_rate = Some(frame_rate);
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid frame count: {0}. Must be 8N+1 (e.g., 9, 17, 25, ...)")]
    InvalidFrameCount(usize),
    #[error("Invalid dimensions: height={height}, width={width}. Must be multiple of 32")]
    InvalidDimensions { height: usize, width: usize },
}
