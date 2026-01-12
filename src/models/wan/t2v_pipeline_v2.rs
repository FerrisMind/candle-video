//! Wan T2V Pipeline with concrete types.
//!
//! This is a simplified pipeline using concrete structs instead of trait objects.
//! Based on diffusers WanPipeline with FlowMatchEulerDiscreteScheduler.

use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use thiserror::Error;

use crate::interfaces::flow_match_scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
};
use crate::interfaces::scheduler_mixin::SchedulerMixin;

use super::transformer_wan::WanTransformer3DModel;
use super::vae::AutoencoderKLWan;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum WanPipelineError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("scheduler error: {0}")]
    Scheduler(String),
}

pub type Result<T> = std::result::Result<T, WanPipelineError>;

// =============================================================================
// Pipeline Output
// =============================================================================

/// Output from the Wan pipeline.
#[derive(Debug)]
pub struct WanPipelineOutput {
    /// Generated video frames tensor [B, C, F, H, W] or latents
    pub frames: Tensor,
}

/// Output type selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputType {
    /// Return latents without VAE decoding
    Latent,
    /// Return decoded video tensor
    Tensor,
}

// =============================================================================
// Text Encoder Trait (minimal interface)
// =============================================================================

/// Minimal text encoder interface for Wan pipeline.
///
/// Implement this for your T5/UMT5 encoder.
pub trait TextEncoder: Send + Sync {
    /// Encode text prompts to embeddings.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `attention_mask` - Attention mask [batch, seq_len]
    ///
    /// # Returns
    /// Hidden states [batch, seq_len, hidden_dim]
    fn encode(&mut self, input_ids: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor>;

    /// Get the data type of the encoder.
    fn dtype(&self) -> DType;
}

/// Tokenized output type alias.
pub type TokenizerOutput = (Vec<Vec<u32>>, Vec<Vec<u8>>);

/// Minimal tokenizer interface.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text prompts.
    ///
    /// # Returns
    /// (input_ids, attention_mask) both as Vec<Vec<u32/u8>>
    fn encode(&self, texts: &[String], max_len: usize) -> Result<TokenizerOutput>;
}

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Configuration for Wan T2V pipeline.
#[derive(Debug, Clone)]
pub struct WanPipelineConfig {
    /// Guidance scale for classifier-free guidance
    pub guidance_scale: f64,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Maximum sequence length for text encoder
    pub max_sequence_length: usize,
    /// Flow shift for scheduler (5.0 for 720p, 3.0 for 480p)
    pub flow_shift: f32,
}

impl Default for WanPipelineConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 5.0,
            num_inference_steps: 50,
            max_sequence_length: 512,
            flow_shift: 5.0, // 720p default
        }
    }
}

impl WanPipelineConfig {
    /// Preset for 720p generation.
    pub fn preset_720p() -> Self {
        Self {
            flow_shift: 5.0,
            ..Default::default()
        }
    }

    /// Preset for 480p generation.
    pub fn preset_480p() -> Self {
        Self {
            flow_shift: 3.0,
            ..Default::default()
        }
    }
}

// =============================================================================
// Wan T2V Pipeline
// =============================================================================

/// Wan Text-to-Video Pipeline with concrete types.
///
/// Uses:
/// - `WanTransformer3DModel` for denoising
/// - `AutoencoderKLWan` for VAE decoding
/// - `FlowMatchEulerDiscreteScheduler` for timestep scheduling
/// - Generic text encoder (T5/UMT5)
pub struct WanT2VPipeline<T: TextEncoder, K: Tokenizer> {
    /// Transformer model
    pub transformer: WanTransformer3DModel,
    /// VAE model
    pub vae: AutoencoderKLWan,
    /// Scheduler
    pub scheduler: FlowMatchEulerDiscreteScheduler,
    /// Text encoder
    pub text_encoder: T,
    /// Tokenizer
    pub tokenizer: K,
    /// Device
    pub device: Device,
    /// Pipeline config
    pub config: WanPipelineConfig,
}

impl<T: TextEncoder, K: Tokenizer> WanT2VPipeline<T, K> {
    /// Create a new Wan T2V pipeline.
    pub fn new(
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        text_encoder: T,
        tokenizer: K,
        device: Device,
        config: WanPipelineConfig,
    ) -> Result<Self> {
        // Create scheduler with flow shift
        let scheduler_config = FlowMatchEulerDiscreteSchedulerConfig {
            shift: config.flow_shift,
            use_dynamic_shifting: false,
            ..Default::default()
        };
        let scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config)
            .map_err(|e| WanPipelineError::Scheduler(e.to_string()))?;

        Ok(Self {
            transformer,
            vae,
            scheduler,
            text_encoder,
            tokenizer,
            device,
            config,
        })
    }

    /// Get VAE scale factors.
    pub fn vae_scale_factor_spatial(&self) -> usize {
        self.vae.scale_factor_spatial()
    }

    pub fn vae_scale_factor_temporal(&self) -> usize {
        self.vae.scale_factor_temporal()
    }

    /// Encode text prompt to embeddings.
    fn encode_prompt(
        &mut self,
        prompt: &str,
        negative_prompt: Option<&str>,
        do_cfg: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let max_len = self.config.max_sequence_length;

        // Tokenize prompt
        let (input_ids, attn_mask) = self.tokenizer.encode(&[prompt.to_string()], max_len)?;

        // Convert to tensors
        let input_ids_t = Tensor::from_vec(
            input_ids[0].iter().map(|&v| v as i64).collect::<Vec<_>>(),
            (1, max_len),
            &self.device,
        )?;
        let attn_mask_t = Tensor::from_vec(
            attn_mask[0].iter().map(|&v| v as i64).collect::<Vec<_>>(),
            (1, max_len),
            &self.device,
        )?;

        // Encode
        let prompt_embeds = self.text_encoder.encode(&input_ids_t, &attn_mask_t)?;

        // Negative prompt for CFG
        let negative_embeds = if do_cfg {
            let neg = negative_prompt.unwrap_or("");
            let (neg_ids, neg_mask) = self.tokenizer.encode(&[neg.to_string()], max_len)?;

            let neg_ids_t = Tensor::from_vec(
                neg_ids[0].iter().map(|&v| v as i64).collect::<Vec<_>>(),
                (1, max_len),
                &self.device,
            )?;
            let neg_mask_t = Tensor::from_vec(
                neg_mask[0].iter().map(|&v| v as i64).collect::<Vec<_>>(),
                (1, max_len),
                &self.device,
            )?;

            Some(self.text_encoder.encode(&neg_ids_t, &neg_mask_t)?)
        } else {
            None
        };

        Ok((prompt_embeds, negative_embeds))
    }

    /// Prepare initial latents.
    fn prepare_latents(
        &self,
        batch_size: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let num_channels = self.transformer.in_channels();
        let num_latent_frames = (num_frames - 1) / self.vae_scale_factor_temporal() + 1;
        let latent_h = height / self.vae_scale_factor_spatial();
        let latent_w = width / self.vae_scale_factor_spatial();

        let shape = [
            batch_size,
            num_channels,
            num_latent_frames,
            latent_h,
            latent_w,
        ];
        randn_tensor(&shape, &self.device, seed)
    }

    /// Run the pipeline.
    ///
    /// # Arguments
    /// * `prompt` - Text prompt for generation
    /// * `negative_prompt` - Optional negative prompt for CFG
    /// * `height` - Output video height (must be divisible by 16)
    /// * `width` - Output video width (must be divisible by 16)
    /// * `num_frames` - Number of output frames
    /// * `output_type` - Whether to return latents or decoded video
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    /// Generated video tensor or latents
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        prompt: &str,
        negative_prompt: Option<&str>,
        height: usize,
        width: usize,
        num_frames: usize,
        output_type: OutputType,
        seed: Option<u64>,
    ) -> Result<WanPipelineOutput> {
        // Validate inputs
        if !height.is_multiple_of(16) || !width.is_multiple_of(16) {
            return Err(WanPipelineError::InvalidArgument(format!(
                "height ({}) and width ({}) must be divisible by 16",
                height, width
            )));
        }

        // Adjust num_frames to be compatible with VAE
        let vae_t = self.vae_scale_factor_temporal();
        let num_frames = if num_frames % vae_t != 1 {
            (num_frames / vae_t) * vae_t + 1
        } else {
            num_frames
        };

        let do_cfg = self.config.guidance_scale > 1.0;
        let transformer_dtype = self.transformer.dtype();

        // 1. Encode prompt
        let (prompt_embeds, negative_embeds) =
            self.encode_prompt(prompt, negative_prompt, do_cfg)?;
        let prompt_embeds = prompt_embeds.to_dtype(transformer_dtype)?;
        let negative_embeds = negative_embeds
            .map(|e| e.to_dtype(transformer_dtype))
            .transpose()?;

        // 2. Set up scheduler
        SchedulerMixin::set_timesteps(
            &mut self.scheduler,
            self.config.num_inference_steps,
            &self.device,
        )
        .map_err(|e| WanPipelineError::Scheduler(e.to_string()))?;
        let timesteps: Vec<f64> = SchedulerMixin::timesteps(&self.scheduler).to_vec();

        // 3. Prepare latents
        let mut latents = self.prepare_latents(1, num_frames, height, width, seed)?;

        // 4. Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            let latent_input = latents.to_dtype(transformer_dtype)?;

            // Create timestep tensor
            let timestep = Tensor::from_vec(vec![t as f32], (1,), &self.device)?;

            // Conditional forward pass
            let noise_pred_cond = self.transformer.forward(
                &latent_input,
                &timestep,
                &prompt_embeds,
                None, // No image conditioning for T2V
                false,
            )?;
            // Extract tensor from Result<Transformer2DModelOutput, Tensor>
            let noise_pred_cond = match noise_pred_cond {
                Ok(output) => output.sample,
                Err(tensor) => tensor,
            };

            // CFG: unconditional + guidance
            let noise_pred = if do_cfg {
                let neg = negative_embeds.as_ref().ok_or_else(|| {
                    WanPipelineError::InvalidArgument(
                        "negative_embeds required for CFG".to_string(),
                    )
                })?;

                let noise_pred_uncond =
                    self.transformer
                        .forward(&latent_input, &timestep, neg, None, false)?;
                let noise_pred_uncond = match noise_pred_uncond {
                    Ok(output) => output.sample,
                    Err(tensor) => tensor,
                };

                // noise_pred = uncond + scale * (cond - uncond)
                let diff = noise_pred_cond.sub(&noise_pred_uncond)?;
                let scaled = diff.affine(self.config.guidance_scale, 0.0)?;
                noise_pred_uncond.add(&scaled)?
            } else {
                noise_pred_cond
            };

            // Scheduler step
            let step_output = SchedulerMixin::step(&mut self.scheduler, &noise_pred, t, &latents)
                .map_err(|e| WanPipelineError::Scheduler(e.to_string()))?;
            latents = step_output.prev_sample;

            // Progress callback could go here
            if i % 10 == 0 {
                eprintln!("Step {}/{}", i + 1, timesteps.len());
            }
        }

        // 5. Decode or return latents
        let output = if output_type == OutputType::Latent {
            latents
        } else {
            // Denormalize latents: latents * std + mean
            let latents = self.vae.denormalize_latents(&latents)?;

            // Decode
            self.vae.decode(&latents)?
        };

        Ok(WanPipelineOutput { frames: output })
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Generate random tensor with optional seed using Box-Muller transform.
fn randn_tensor(shape: &[usize], device: &Device, seed: Option<u64>) -> Result<Tensor> {
    use rand::Rng;

    let numel: usize = shape.iter().product();
    let mut data = Vec::<f32>::with_capacity(numel);

    // Box-Muller transform for normal distribution
    fn box_muller<R: Rng>(rng: &mut R) -> (f64, f64) {
        let u1: f64 = rng.gen_range(1e-10..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    match seed {
        Some(seed) => {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut i = 0;
            while i < numel {
                let (z0, z1) = box_muller(&mut rng);
                data.push(z0 as f32);
                i += 1;
                if i < numel {
                    data.push(z1 as f32);
                    i += 1;
                }
            }
        }
        None => {
            let mut rng = rand::thread_rng();
            let mut i = 0;
            while i < numel {
                let (z0, z1) = box_muller(&mut rng);
                data.push(z0 as f32);
                i += 1;
                if i < numel {
                    data.push(z1 as f32);
                    i += 1;
                }
            }
        }
    }

    let t = Tensor::from_vec(data, Shape::from_dims(shape), device)?;
    Ok(t)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_presets() {
        let cfg_720p = WanPipelineConfig::preset_720p();
        assert_eq!(cfg_720p.flow_shift, 5.0);

        let cfg_480p = WanPipelineConfig::preset_480p();
        assert_eq!(cfg_480p.flow_shift, 3.0);
    }

    #[test]
    fn test_randn_tensor_seeded() {
        let device = Device::Cpu;
        let t1 = randn_tensor(&[2, 3], &device, Some(42)).unwrap();
        let t2 = randn_tensor(&[2, 3], &device, Some(42)).unwrap();

        // Same seed should produce same values
        let v1: Vec<f32> = t1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = t2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }
}
