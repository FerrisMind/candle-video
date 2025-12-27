//! Text-to-Video Pipeline for LTX-Video
//!
//! This module implements the complete text-to-video generation pipeline,
//! integrating all components: T5 text encoder, DiT (Diffusion Transformer),
//! VAE (Video Autoencoder), and Rectified Flow scheduler.
//!
//! # Architecture Overview
//!
//! ```text
//! Text Prompt → T5 Encoder → Text Embeddings
//!                                   ↓
//!            Random Noise → DiT Denoising Loop → Denoised Latents
//!                                   ↓
//!                          VAE Decoder → Video Frames
//! ```
//!
//! # Features
//!
//! - **Classifier-Free Guidance (CFG)**: Supports guided generation with
//!   configurable guidance scale for better prompt adherence
//! - **Progress Callbacks**: Real-time progress reporting during generation
//! - **Mock Generation**: Testing support without loaded model weights
//! - **Configurable Parameters**: Adjustable inference steps, guidance scale, etc.
//!
//! # Example
//!
//! ```rust,ignore
//! use candle_video::pipeline::{TextToVideoPipeline, PipelineConfig};
//! use candle_video::text_encoder::T5TextEncoderWrapper;
//!
//! // Create pipeline
//! let pipeline = TextToVideoPipeline::new(device, config)?;
//!
//! // Encode text
//! let text_emb = encoder.encode("A sunset over the ocean")?;
//!
//! // Generate video
//! let video = pipeline.generate(&text_emb, &inference_config)?;
//! ```

use crate::config::{DitConfig, InferenceConfig, SchedulerConfig, VaeConfig};
use crate::dit::Transformer3DModel;
use crate::scheduler::RectifiedFlowScheduler;
use crate::vae::{VaeDecoder, VaeEncoder};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use image::{DynamicImage, imageops::FilterType};
use std::path::Path;

// =============================================================================
// Latent Manipulation Utilities
// =============================================================================

/// Pack latents from 5D to 3D for transformer input
/// Matches diffusers `_pack_latents`: [B, C, F, H, W] -> [B, seq_len, C * patch_t * patch_h * patch_w]
pub fn pack_latents(latents: &Tensor, patch_size: usize, patch_size_t: usize) -> Result<Tensor> {
    let (batch_size, num_channels, num_frames, height, width) = latents.dims5()?;

    let post_patch_frames = num_frames / patch_size_t;
    let post_patch_height = height / patch_size;
    let post_patch_width = width / patch_size;

    // Reshape to [B, C, F//pt, pt, H//p, p, W//p, p] - need to do in steps for Candle 6D limit
    // Step 1: temporal patching - [B, C, F, H, W] -> [B, C, F//pt, pt, H*W]
    let x = latents.reshape((
        batch_size,
        num_channels,
        post_patch_frames,
        patch_size_t,
        height * width,
    ))?;
    // -> [B, C*pt, F//pt, H*W]
    let x = x.permute((0, 1, 3, 2, 4))?;
    let x = x.reshape((
        batch_size,
        num_channels * patch_size_t,
        post_patch_frames,
        height,
        width,
    ))?;

    // Step 2: spatial patching per frame - flatten batch*time for 4D operations
    let c_new = num_channels * patch_size_t;
    let x = x.reshape((batch_size * post_patch_frames, c_new, height, width))?;

    // [B*F', C*pt, H, W] -> [B*F', C*pt, H//p, p, W//p, p]
    let x = x.reshape((
        batch_size * post_patch_frames,
        c_new,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    ))?;
    // -> [B*F', C*pt*p*p, H//p, W//p]
    let x = x.permute((0, 1, 3, 5, 2, 4))?;
    let c_packed = c_new * patch_size * patch_size;
    let x = x.reshape((
        batch_size * post_patch_frames,
        c_packed,
        post_patch_height,
        post_patch_width,
    ))?;

    // [B*F', C_packed, H', W'] -> [B, F'*H'*W', C_packed]
    let x = x.reshape((
        batch_size,
        post_patch_frames,
        c_packed,
        post_patch_height * post_patch_width,
    ))?;
    let x = x.permute((0, 1, 3, 2))?;
    let seq_len = post_patch_frames * post_patch_height * post_patch_width;
    x.reshape((batch_size, seq_len, c_packed))
}

/// Unpack latents from 3D to 5D for VAE decode
/// Matches diffusers `_unpack_latents`: [B, seq_len, D] -> [B, C, F, H, W]
///
/// Uses multi-step approach to stay within Candle's 6D tensor limit
pub fn unpack_latents(
    latents: &Tensor,
    num_frames: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    patch_size_t: usize,
) -> Result<Tensor> {
    let batch_size = latents.dim(0)?;
    let d = latents.dim(2)?;
    let num_channels = d / (patch_size_t * patch_size * patch_size);

    let out_frames = num_frames * patch_size_t;
    let out_height = height * patch_size;
    let out_width = width * patch_size;

    // Step 1: [B, seq_len, D] -> [B, F', H', W', D]
    let x = latents.reshape((batch_size, num_frames, height, width, d))?;

    // Step 2: Split D into [C, pt, p, p] - but need to do in steps
    // [B, F', H', W', C*pt*p*p] -> [B, F', H', W', C*pt, p*p]
    let c_pt = num_channels * patch_size_t;
    let p_sq = patch_size * patch_size;
    let x = x.reshape((batch_size, num_frames, height, width, c_pt, p_sq))?;

    // Step 3: Unfold spatial patches
    // [B, F', H', W', C*pt, p*p] -> [B, F', H', W', C*pt, p, p]
    // Flatten batch dims for 6D operation then reshape
    let x = x.reshape((
        batch_size * num_frames * height * width,
        c_pt,
        patch_size,
        patch_size,
    ))?;

    // Now we have [B*F'*H'*W', C*pt, p, p]
    // Unfold back to spatial: [B*F', H'*W', C*pt, p, p] -> [B*F', C*pt, H', p, W', p]
    let x = x.reshape((
        batch_size * num_frames,
        height * width,
        c_pt,
        patch_size,
        patch_size,
    ))?;
    let x = x.reshape((
        batch_size * num_frames,
        height,
        width,
        c_pt,
        patch_size,
        patch_size,
    ))?;
    // Permute: [B*F', H', W', C*pt, p, p] -> [B*F', C*pt, H', p, W', p]
    let x = x.permute((0, 3, 1, 4, 2, 5))?;
    // Flatten spatial: [B*F', C*pt, H'*p, W'*p]
    let x = x.reshape((batch_size * num_frames, c_pt, out_height, out_width))?;

    // Step 4: Unfold temporal patches
    // [B*F', C*pt, H, W] -> [B, F', C*pt, H, W]
    let x = x.reshape((batch_size, num_frames, c_pt, out_height, out_width))?;
    // -> [B, F', C, pt, H, W]
    let x = x.reshape((
        batch_size,
        num_frames,
        num_channels,
        patch_size_t,
        out_height,
        out_width,
    ))?;
    // Permute: [B, F', C, pt, H, W] -> [B, C, F', pt, H, W]
    let x = x.permute((0, 2, 1, 3, 4, 5))?;
    // Flatten temporal: [B, C, F'*pt, H, W]
    x.reshape((batch_size, num_channels, out_frames, out_height, out_width))
}

/// Normalize latents before transformer
/// Matches diffusers: (latents - mean) * scaling_factor / std
pub fn normalize_latents(
    latents: &Tensor,
    latents_mean: &Tensor,
    latents_std: &Tensor,
    scaling_factor: f64,
) -> Result<Tensor> {
    let device = latents.device();
    let dtype = latents.dtype();

    // Reshape mean/std from [C] to [1, C, 1, 1, 1] for broadcasting
    let mean = latents_mean
        .reshape((1, (), 1, 1, 1))?
        .to_device(device)?
        .to_dtype(dtype)?;
    let std = latents_std
        .reshape((1, (), 1, 1, 1))?
        .to_device(device)?
        .to_dtype(dtype)?;
    let scale = Tensor::new(&[scaling_factor as f32], device)?.to_dtype(dtype)?;

    // (latents - mean) * scaling_factor / std
    latents
        .broadcast_sub(&mean)?
        .broadcast_mul(&scale)?
        .broadcast_div(&std)
}

/// Denormalize latents before VAE decode
/// Matches diffusers: latents * std / scaling_factor + mean
pub fn denormalize_latents(
    latents: &Tensor,
    latents_mean: &Tensor,
    latents_std: &Tensor,
    scaling_factor: f64,
) -> Result<Tensor> {
    let device = latents.device();
    let dtype = latents.dtype();

    // Reshape mean/std from [C] to [1, C, 1, 1, 1] for broadcasting
    let mean = latents_mean
        .reshape((1, (), 1, 1, 1))?
        .to_device(device)?
        .to_dtype(dtype)?;
    let std = latents_std
        .reshape((1, (), 1, 1, 1))?
        .to_device(device)?
        .to_dtype(dtype)?;
    let scale = Tensor::new(&[scaling_factor as f32], device)?.to_dtype(dtype)?;

    // latents * std / scaling_factor + mean
    latents
        .broadcast_mul(&std)?
        .broadcast_div(&scale)?
        .broadcast_add(&mean)
}

// =============================================================================
// CFG-Star Rescale
// =============================================================================

/// Apply CFG-star rescale to unconditional prediction
///
/// Matches Python pipeline_ltx_video.py lines 1227-1240
///
/// This rescales the unconditional prediction based on the ratio of
/// standard deviations between conditional and unconditional predictions,
/// improving guidance quality especially at high CFG scales.
///
/// # Formula
/// `uncond_rescaled = cond + (uncond - cond) * (std(cond) / std(uncond))`
///
/// # Arguments
/// * `noise_pred_text` - Conditional (text-guided) prediction [B, ...]
/// * `noise_pred_uncond` - Unconditional prediction [B, ...]
///
/// # Returns
/// Rescaled unconditional prediction
pub fn apply_cfg_star_rescale(
    noise_pred_text: &Tensor,
    noise_pred_uncond: &Tensor,
) -> Result<Tensor> {
    // Compute std over all dimensions except batch
    // Flatten to [B, -1] for easier std computation
    let batch_size = noise_pred_text.dim(0)?;
    let text_flat = noise_pred_text.flatten(1, noise_pred_text.rank() - 1)?;
    let uncond_flat = noise_pred_uncond.flatten(1, noise_pred_uncond.rank() - 1)?;

    // Compute variance = mean((x - mean(x))^2)
    let text_mean = text_flat.mean_keepdim(1)?;
    let text_centered = text_flat.broadcast_sub(&text_mean)?;
    let text_var = text_centered.sqr()?.mean_keepdim(1)?;
    let text_std = text_var.sqrt()?;

    let uncond_mean = uncond_flat.mean_keepdim(1)?;
    let uncond_centered = uncond_flat.broadcast_sub(&uncond_mean)?;
    let uncond_var = uncond_centered.sqr()?.mean_keepdim(1)?;
    let uncond_std = uncond_var.sqrt()?;

    // Compute scale ratio: std(cond) / std(uncond)
    // Add small epsilon to avoid division by zero
    let eps = Tensor::full(1e-8f32, uncond_std.shape(), uncond_std.device())?
        .to_dtype(uncond_std.dtype())?;
    let scale_ratio = text_std.broadcast_div(&uncond_std.broadcast_add(&eps)?)?;

    // Reshape scale_ratio to match original tensor shape for broadcasting
    // [B, 1] -> [B, 1, 1, 1, 1] for 5D tensors
    let mut shape = vec![batch_size];
    shape.extend(std::iter::repeat_n(1, noise_pred_uncond.rank() - 1));
    let scale_ratio = scale_ratio.reshape(shape)?;

    // uncond_rescaled = cond + (uncond - cond) * scale_ratio
    let diff = noise_pred_uncond.sub(noise_pred_text)?;
    let scaled_diff = diff.broadcast_mul(&scale_ratio)?;
    noise_pred_text.add(&scaled_diff)
}

// =============================================================================
// Tone Mapping
// =============================================================================

/// Apply tone mapping compression to latents
///
/// Matches Python pipeline_ltx_video.py tone_map_latents
///
/// Compresses the dynamic range of latents to prevent over-saturation,
/// particularly useful with high-range distilled models.
///
/// # Formula
/// `mapped = latents / max(1, max(abs(latents)) * compression_ratio)`
///
/// # Arguments
/// * `latents` - Input latents [B, C, T, H, W]
/// * `compression_ratio` - Compression factor (typically 0.95-1.0)
///
/// # Returns
/// Tone-mapped latents
pub fn tone_map_latents(latents: &Tensor, compression_ratio: f64) -> Result<Tensor> {
    if compression_ratio >= 1.0 {
        // No compression needed
        return Ok(latents.clone());
    }

    let device = latents.device();
    let dtype = latents.dtype();

    // Find max absolute value across all dimensions
    let abs_latents = latents.abs()?;
    let max_val = abs_latents.max(candle_core::D::Minus1)?;
    let max_val = max_val.max(candle_core::D::Minus1)?;
    let max_val = max_val.max(candle_core::D::Minus1)?;
    let max_val = max_val.max(candle_core::D::Minus1)?; // Now [B]

    // scale = max(1, max_val * compression_ratio)
    let one = Tensor::ones(max_val.shape(), dtype, device)?;
    let scaled_max = max_val.affine(compression_ratio, 0.0)?;
    let scale = scaled_max.maximum(&one)?;

    // Reshape scale for broadcasting: [B] -> [B, 1, 1, 1, 1]
    let batch_size = latents.dim(0)?;
    let scale = scale.reshape((batch_size, 1, 1, 1, 1))?;

    // mapped = latents / scale
    latents.broadcast_div(&scale)
}

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Complete configuration for the text-to-video pipeline
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    /// DiT (Diffusion Transformer) configuration
    pub dit: DitConfig,
    /// VAE configuration
    pub vae: VaeConfig,
    /// Scheduler configuration
    pub scheduler: SchedulerConfig,
}

// =============================================================================
// TextToVideoPipeline
// =============================================================================

// =============================================================================
// Image-to-Video Conditioning
// =============================================================================

/// Conditioning item for image-to-video generation
///
/// Represents a media item (image or video frames) that will be used as conditioning
/// for video generation. The conditioning can be applied at different frame positions
/// with configurable strength.
#[derive(Debug, Clone)]
pub struct ConditioningItem {
    /// Latent tensor [B, C, F, H, W] - already encoded by VAE
    pub latents: Tensor,
    /// Frame number where conditioning starts (0 = first frame)
    pub frame_number: usize,
    /// Conditioning strength (1.0 = full conditioning, 0.0 = no conditioning)
    pub strength: f32,
}

impl ConditioningItem {
    /// Create a new conditioning item
    pub fn new(latents: Tensor, frame_number: usize, strength: f32) -> Self {
        Self {
            latents,
            frame_number,
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

/// Add timestep-dependent noise to conditioning latents
///
/// This helps with motion continuity, especially when conditioned on a single frame.
/// Matches Python: add_noise_to_image_conditioning_latents
///
/// # Arguments
/// * `t` - Current timestep value
/// * `init_latents` - Original conditioning latents [B, seq, C]
/// * `latents` - Current (possibly noisy) latents [B, seq, C]
/// * `noise_scale` - Scale factor for noise addition
/// * `conditioning_mask` - Mask where 1.0 = conditioning token [B, seq]
pub fn add_noise_to_conditioning_latents(
    t: f64,
    init_latents: &Tensor,
    latents: &Tensor,
    noise_scale: f64,
    conditioning_mask: &Tensor,
) -> Result<Tensor> {
    let eps = 1e-6;
    let device = latents.device();
    let dtype = latents.dtype();

    // Generate noise with same shape as latents
    let noise = Tensor::randn(0f32, 1.0, latents.shape(), device)?.to_dtype(dtype)?;

    // need_to_noise = conditioning_mask > (1.0 - eps)
    let threshold =
        Tensor::full((1.0 - eps) as f32, conditioning_mask.shape(), device)?.to_dtype(dtype)?;
    let need_to_noise = conditioning_mask.gt(&threshold)?;

    // noised_latents = init_latents + noise_scale * noise * (t^2)
    let t_squared = (t * t) as f32;
    let scale = (noise_scale * t_squared as f64) as f32;
    let scaled_noise = noise.affine(scale as f64, 0.0)?;
    let noised_latents = init_latents.add(&scaled_noise)?;
    // Expand mask for broadcasting: [B, seq] -> [B, seq, 1]
    let mask_expanded = need_to_noise.unsqueeze(2)?.to_dtype(dtype)?;

    // result = where(need_to_noise, noised_latents, latents)
    let diff = noised_latents.sub(latents)?;
    let masked_diff = diff.broadcast_mul(&mask_expanded)?;
    latents.add(&masked_diff)
}

/// Compute per-token timestep from conditioning mask
///
/// For conditioning tokens, the effective timestep is (1.0 - conditioning_strength).
/// This means fully conditioned tokens (strength=1.0) have timestep=0 (no noise).
///
/// # Arguments
/// * `base_timestep` - Global timestep for the current step
/// * `conditioning_mask` - Mask with conditioning strength per token [B, seq]
///
/// # Returns
/// Per-token timesteps [B, seq]
pub fn compute_token_timesteps(base_timestep: f64, conditioning_mask: &Tensor) -> Result<Tensor> {
    let device = conditioning_mask.device();
    let dtype = conditioning_mask.dtype();

    // effective_timestep = min(base_timestep, 1.0 - conditioning_mask)
    let base_t =
        Tensor::full(base_timestep as f32, conditioning_mask.shape(), device)?.to_dtype(dtype)?;
    let one = Tensor::ones(conditioning_mask.shape(), dtype, device)?;
    let cond_ceiling = one.sub(conditioning_mask)?;

    // Element-wise minimum
    base_t.minimum(&cond_ceiling)
}

/// Apply denoising step with conditioning mask
///
/// Only denoise tokens where current timestep is below their conditioning ceiling.
/// Hard-conditioning tokens (mask=1.0) are never denoised.
///
/// # Arguments
/// * `denoised` - Denoised latents from scheduler step [B, seq, C]
/// * `original` - Original latents before denoising [B, seq, C]
/// * `t` - Current timestep
/// * `conditioning_mask` - Per-token conditioning strength [B, seq]
///
/// # Returns
/// Latents with selective denoising applied
pub fn apply_conditioning_mask(
    denoised: &Tensor,
    original: &Tensor,
    t: f64,
    conditioning_mask: &Tensor,
) -> Result<Tensor> {
    let t_eps = 1e-6;
    let device = original.device();
    let dtype = original.dtype();

    // tokens_to_denoise = (t - eps) < (1.0 - conditioning_mask)
    let t_val =
        Tensor::full((t - t_eps) as f32, conditioning_mask.shape(), device)?.to_dtype(dtype)?;
    let one = Tensor::ones(conditioning_mask.shape(), dtype, device)?;
    let threshold = one.sub(conditioning_mask)?;
    let tokens_to_denoise = t_val.lt(&threshold)?;

    // Expand mask for broadcasting: [B, seq] -> [B, seq, 1]
    let mask_expanded = tokens_to_denoise.unsqueeze(2)?.to_dtype(dtype)?;

    // result = where(tokens_to_denoise, denoised, original)
    let diff = denoised.sub(original)?;
    let masked_diff = diff.broadcast_mul(&mask_expanded)?;
    original.add(&masked_diff)
}

// =============================================================================
// TextToVideoPipeline
// =============================================================================

/// Text-to-Video pipeline orchestrating all components
///
/// This pipeline coordinates the text encoder, DiT model, VAE decoder,
/// and scheduler to generate videos from text prompts.
pub struct TextToVideoPipeline {
    device: Device,
    dit_config: DitConfig,
    vae_config: VaeConfig,
    scheduler: RectifiedFlowScheduler,
    dit_model: Option<Transformer3DModel>,
    vae_decoder: Option<VaeDecoder>,
    vae_encoder: Option<VaeEncoder>,
    latents_mean: Option<Tensor>,
    latents_std: Option<Tensor>,
}

impl TextToVideoPipeline {
    /// Create a new pipeline from configuration
    ///
    /// # Arguments
    /// * `device` - Computation device (CPU/CUDA)
    /// * `config` - Pipeline configuration containing DiT, VAE, and scheduler configs
    pub fn new(device: Device, config: PipelineConfig) -> Result<Self> {
        let scheduler = RectifiedFlowScheduler::new(config.scheduler);

        Ok(Self {
            device,
            dit_config: config.dit,
            vae_config: config.vae,
            scheduler,
            dit_model: None,
            vae_decoder: None,
            vae_encoder: None,
            latents_mean: None,
            latents_std: None,
        })
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get reference to the scheduler
    pub fn scheduler(&self) -> &RectifiedFlowScheduler {
        &self.scheduler
    }

    /// Get mutable reference to the scheduler
    pub fn scheduler_mut(&mut self) -> &mut RectifiedFlowScheduler {
        &mut self.scheduler
    }

    /// Get reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get reference to DiT config
    pub fn dit_config(&self) -> &DitConfig {
        &self.dit_config
    }

    /// Get reference to VAE config
    pub fn vae_config(&self) -> &VaeConfig {
        &self.vae_config
    }

    // =========================================================================
    // Model Loading
    // =========================================================================

    fn stats_key(prefix: Option<&str>, name: &str) -> String {
        match prefix {
            Some(p) if !p.is_empty() => format!("{}.{}", p, name),
            _ => name.to_string(),
        }
    }

    /// Load DiT model weights from a safetensors file.
    pub fn load_dit(&mut self, model_path: impl AsRef<Path>, dtype: DType) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], dtype, &self.device)?
        };
        let vb = vb.pp("model.diffusion_model");
        let model = Transformer3DModel::new(vb, &self.dit_config)?;
        self.dit_model = Some(model);
        Ok(())
    }

    /// Load VAE decoder weights from a safetensors file.
    pub fn load_vae_decoder(
        &mut self,
        model_path: impl AsRef<Path>,
        prefix: &str,
        dtype: DType,
    ) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], dtype, &self.device)?
        };
        let vb = vb.pp(prefix);
        let decoder = VaeDecoder::new(vb)?;
        self.vae_decoder = Some(decoder);
        Ok(())
    }

    /// Load VAE encoder weights from a safetensors file.
    pub fn load_vae_encoder(
        &mut self,
        model_path: impl AsRef<Path>,
        prefix: &str,
        dtype: DType,
    ) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], dtype, &self.device)?
        };
        let vb = vb.pp(prefix);
        let encoder = VaeEncoder::new(vb)?;
        self.vae_encoder = Some(encoder);
        Ok(())
    }

    /// Load per-channel latent statistics (mean/std) for denormalization.
    pub fn load_latent_stats(
        &mut self,
        model_path: impl AsRef<Path>,
        prefix: Option<&str>,
        dtype: DType,
    ) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], dtype, &self.device)?
        };
        let channels = self.vae_config.latent_channels;
        let mean_key = Self::stats_key(prefix, "per_channel_statistics.mean-of-means");
        let std_key = Self::stats_key(prefix, "per_channel_statistics.std-of-means");

        let latents_mean = match vb.get((channels,), mean_key.as_str()) {
            Ok(tensor) => tensor,
            Err(_) => Tensor::zeros((channels,), dtype, &self.device)?,
        };
        let latents_std = match vb.get((channels,), std_key.as_str()) {
            Ok(tensor) => tensor,
            Err(_) => Tensor::ones((channels,), dtype, &self.device)?,
        };

        self.latents_mean = Some(latents_mean);
        self.latents_std = Some(latents_std);
        Ok(())
    }

    // =========================================================================
    // Configuration Mutators
    // =========================================================================

    /// Set guidance scale
    pub fn set_guidance_scale(&mut self, scale: f64) {
        self.scheduler = RectifiedFlowScheduler::new(SchedulerConfig {
            guidance_scale: scale,
            num_inference_steps: self.scheduler.num_steps(),
            ..SchedulerConfig::default()
        });
    }

    /// Set number of inference steps
    pub fn set_num_inference_steps(&mut self, steps: usize) {
        self.scheduler = RectifiedFlowScheduler::new(SchedulerConfig {
            num_inference_steps: steps,
            guidance_scale: self.scheduler.guidance_scale(),
            ..SchedulerConfig::default()
        });
    }

    // =========================================================================
    // Dimension Calculations
    // =========================================================================

    /// Compute latent dimensions from inference config
    ///
    /// Returns (T_latent, H_latent, W_latent)
    pub fn compute_latent_dims(&self, config: &InferenceConfig) -> (usize, usize, usize) {
        let lat_t = (config.num_frames.saturating_sub(1)) / self.vae_config.temporal_downsample + 1;
        let lat_h = config.height / self.vae_config.spatial_downsample;
        let lat_w = config.width / self.vae_config.spatial_downsample;
        (lat_t, lat_h, lat_w)
    }

    // =========================================================================
    // Noise Initialization
    // =========================================================================

    /// Initialize random noise for diffusion process
    ///
    /// Creates noise tensor with shape (1, C_latent, T_latent, H_latent, W_latent)
    pub fn initialize_noise(&self, config: &InferenceConfig) -> Result<Tensor> {
        let (lat_t, lat_h, lat_w) = self.compute_latent_dims(config);

        Tensor::randn(
            0f32,
            1.0,
            (1, self.vae_config.latent_channels, lat_t, lat_h, lat_w),
            &self.device,
        )
    }

    /// Initialize noise with a specific seed for reproducibility
    ///
    /// Note: True reproducibility depends on the device's RNG implementation
    pub fn initialize_noise_with_seed(
        &self,
        config: &InferenceConfig,
        _seed: u64,
    ) -> Result<Tensor> {
        // Note: candle_core doesn't have seeded RNG yet, so we use standard randn
        // In production, we'd use a seeded generator here
        self.initialize_noise(config)
    }

    // =========================================================================
    // RoPE Coordinate Generation
    // =========================================================================

    /// Generate RoPE indices grid for positional embeddings
    ///
    /// Returns tensor of shape (batch_size, 3, seq_len) containing
    /// normalized (T, H, W) coordinates for each position
    pub fn generate_rope_indices(&self, config: &InferenceConfig) -> Result<Tensor> {
        let (lat_t, lat_h, lat_w) = self.compute_latent_dims(config);
        crate::rope::generate_indices_grid_raw(1, lat_t, lat_h, lat_w, &self.device)
    }

    // =========================================================================
    // CFG (Classifier-Free Guidance) Support
    // =========================================================================

    /// Prepare embeddings for CFG by concatenating negative and positive
    ///
    /// # Arguments
    /// * `positive_emb` - Positive (conditional) text embeddings
    /// * `negative_emb` - Negative (unconditional) text embeddings
    ///
    /// # Returns
    /// Concatenated tensor of shape (2, seq_len, dim) with [negative, positive]
    pub fn prepare_cfg_embeddings(
        &self,
        positive_emb: &Tensor,
        negative_emb: &Tensor,
    ) -> Result<Tensor> {
        // CFG order: [unconditional, conditional] for proper guidance
        Tensor::cat(&[negative_emb, positive_emb], 0)
    }

    /// Apply CFG to batched model output
    ///
    /// # Arguments
    /// * `model_output` - Model output of shape (2, ...) where [0] is unconditional
    ///   and [1] is conditional
    ///
    /// # Returns
    /// Guided output of shape (1, ...)
    pub fn apply_cfg(&self, model_output: &Tensor) -> Result<Tensor> {
        // Split into unconditional and conditional
        let uncond = model_output.i(0)?;
        let cond = model_output.i(1)?;

        // Apply CFG: guided = uncond + guidance_scale * (cond - uncond)
        // Add batch dimension back with unsqueeze
        self.scheduler.apply_cfg(&cond, &uncond)?.unsqueeze(0)
    }

    /// Duplicate latents for CFG (unconditional + conditional batch)
    fn duplicate_for_cfg(&self, latents: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[latents, latents], 0)
    }

    // =========================================================================
    // Timestep Utilities
    // =========================================================================

    /// Create timestep tensor for batch
    ///
    /// # Arguments
    /// * `timestep` - Timestep value
    /// * `batch_size` - Number of samples in batch
    pub fn create_timestep_tensor(&self, timestep: f64, batch_size: usize) -> Result<Tensor> {
        let timesteps = vec![timestep as f32; batch_size];
        Tensor::from_vec(timesteps, (batch_size,), &self.device)
    }

    fn require_dit(&self) -> Result<&Transformer3DModel> {
        self.dit_model
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("DiT model not loaded".to_string()))
    }

    fn require_vae_decoder(&self) -> Result<&VaeDecoder> {
        self.vae_decoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("VAE decoder not loaded".to_string()))
    }

    fn require_latent_stats(&self) -> Result<(&Tensor, &Tensor)> {
        let mean = self
            .latents_mean
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Latent mean not loaded".to_string()))?;
        let std = self
            .latents_std
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Latent std not loaded".to_string()))?;
        Ok((mean, std))
    }

    fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let (mean, std) = self.require_latent_stats()?;
        let channels = self.vae_config.latent_channels;
        let dtype = latents.dtype();

        let mean = mean.reshape((1, channels, 1, 1, 1))?.to_dtype(dtype)?;
        let std = std.reshape((1, channels, 1, 1, 1))?.to_dtype(dtype)?;

        latents.broadcast_mul(&std)?.broadcast_add(&mean)
    }

    // =========================================================================
    // Mock Operations (for testing without loaded models)
    // =========================================================================

    /// Mock DiT forward pass
    ///
    /// Returns velocity prediction with same shape as input latents
    pub fn mock_dit_forward(
        &self,
        latents: &Tensor,
        _text_emb: &Tensor,
        _timestep: f64,
    ) -> Result<Tensor> {
        // Return random velocity prediction
        Tensor::randn(0f32, 0.1, latents.shape(), &self.device)
    }

    /// Mock VAE decode
    ///
    /// Returns video frames tensor
    pub fn mock_vae_decode(&self, latents: &Tensor, config: &InferenceConfig) -> Result<Tensor> {
        let batch_size = latents.dim(0)?;

        // Output: (B, 3, T, H, W) - RGB video
        Tensor::randn(
            0f32,
            1.0,
            (
                batch_size,
                3,
                config.num_frames,
                config.height,
                config.width,
            ),
            &self.device,
        )
    }

    /// Mock single denoising step
    pub fn mock_denoising_step(
        &self,
        latents: &Tensor,
        text_emb: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        // Get timestep for this step
        let timestep = self
            .scheduler
            .timesteps()
            .get(step_idx)
            .copied()
            .unwrap_or(0.5);

        // Mock velocity prediction
        let velocity = self.mock_dit_forward(latents, text_emb, timestep)?;

        // Apply scheduler step (Euler step)
        self.scheduler.step(&velocity, latents, step_idx)
    }

    /// Mock denoising step with CFG
    fn mock_denoising_step_cfg(
        &self,
        latents: &Tensor,
        pos_emb: &Tensor,
        neg_emb: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        let timestep = self
            .scheduler
            .timesteps()
            .get(step_idx)
            .copied()
            .unwrap_or(0.5);

        // Create CFG batch: duplicate latents, concatenate embeddings
        let cfg_latents = self.duplicate_for_cfg(latents)?;
        let cfg_emb = self.prepare_cfg_embeddings(pos_emb, neg_emb)?;

        // Mock batched velocity prediction
        let velocity = self.mock_dit_forward(&cfg_latents, &cfg_emb, timestep)?;

        // Apply CFG (returns tensor with batch dimension)
        let guided_velocity = self.apply_cfg(&velocity)?;

        // Apply scheduler step
        self.scheduler.step(&guided_velocity, latents, step_idx)
    }

    /// Mock full denoising loop with callback
    pub fn mock_denoise_with_callback<F>(
        &self,
        text_emb: &Tensor,
        config: &InferenceConfig,
        mut callback: F,
    ) -> Result<Tensor>
    where
        F: FnMut(usize, &Tensor),
    {
        let mut latents = self.initialize_noise(config)?;
        let num_steps = self.scheduler.num_steps();

        for step in 0..num_steps {
            latents = self.mock_denoising_step(&latents, text_emb, step)?;
            callback(step, &latents);
        }

        Ok(latents)
    }

    /// Mock generate video without CFG
    pub fn mock_generate(&self, text_emb: &Tensor, config: &InferenceConfig) -> Result<Tensor> {
        // Initialize noise
        let mut latents = self.initialize_noise(config)?;

        // Denoising loop
        for step in 0..self.scheduler.num_steps() {
            latents = self.mock_denoising_step(&latents, text_emb, step)?;
        }

        // Decode to video
        self.mock_vae_decode(&latents, config)
    }

    /// Mock generate video with CFG
    pub fn mock_generate_with_cfg(
        &self,
        pos_emb: &Tensor,
        neg_emb: &Tensor,
        config: &InferenceConfig,
    ) -> Result<Tensor> {
        // Initialize noise
        let mut latents = self.initialize_noise(config)?;

        // Denoising loop with CFG
        for step in 0..self.scheduler.num_steps() {
            latents = self.mock_denoising_step_cfg(&latents, pos_emb, neg_emb, step)?;
        }

        // Decode to video
        self.mock_vae_decode(&latents, config)
    }

    /// Mock generate with progress callback
    pub fn mock_generate_with_progress<F>(
        &self,
        text_emb: &Tensor,
        config: &InferenceConfig,
        mut progress_callback: F,
    ) -> Result<Tensor>
    where
        F: FnMut(f32),
    {
        let mut latents = self.initialize_noise(config)?;
        let num_steps = self.scheduler.num_steps();

        for step in 0..num_steps {
            latents = self.mock_denoising_step(&latents, text_emb, step)?;

            // Report progress (0.0 to 1.0)
            let progress = (step + 1) as f32 / num_steps as f32;
            progress_callback(progress);
        }

        self.mock_vae_decode(&latents, config)
    }

    // =========================================================================
    // Full Pipeline (with loaded models - placeholder for now)
    // =========================================================================

    /// Generate video from text embeddings
    ///
    /// This is the main entry point for video generation. Requires models to be loaded.
    ///
    /// # Arguments
    /// * `text_embeddings` - Pre-computed text embeddings from T5 encoder
    /// * `config` - Inference configuration (frames, dimensions, etc.)
    ///
    /// # Note
    /// Currently uses mock operations. Full implementation requires loaded DiT and VAE models.
    pub fn generate(
        &mut self,
        text_embeddings: &Tensor,
        config: &InferenceConfig,
    ) -> Result<Tensor> {
        let dtype = text_embeddings.dtype();

        let (lat_t, lat_h, lat_w) = self.compute_latent_dims(config);
        let mut latents = self.initialize_noise(config)?.to_dtype(dtype)?;

        let sample_shape = (1, self.vae_config.latent_channels, lat_t, lat_h, lat_w);
        self.scheduler.set_timesteps_with_shape(
            self.scheduler.num_steps(),
            Some(sample_shape),
            None,
            None,
        )?;

        let indices_grid = crate::rope::generate_indices_grid_diffusers(
            1,
            lat_t,
            lat_h,
            lat_w,
            config.frame_rate,
            &self.device,
        )?;

        let num_steps = self.scheduler.num_steps();
        for step_idx in 0..num_steps {
            let timestep = self.scheduler.timesteps()[step_idx];
            let timestep_tensor = Tensor::new(&[timestep as f32], &self.device)?.to_dtype(dtype)?;

            let velocity = self.require_dit()?.forward(
                &latents,
                &indices_grid,
                Some(text_embeddings),
                &timestep_tensor,
                None,
                None,
                None,
                None,
            )?;

            latents = self.scheduler.step(&velocity, &latents, step_idx)?;
        }

        let latents = self.denormalize_latents(&latents)?;
        let decode_timestep = if self.vae_config.timestep_conditioning {
            Some(0.05)
        } else {
            None
        };

        self.require_vae_decoder()?
            .decode(&latents, decode_timestep)
    }

    /// Generate video with CFG
    ///
    /// # Arguments
    /// * `positive_embeddings` - Positive (conditional) text embeddings
    /// * `negative_embeddings` - Negative (unconditional) text embeddings
    /// * `config` - Inference configuration
    pub fn generate_with_cfg(
        &mut self,
        positive_embeddings: &Tensor,
        negative_embeddings: &Tensor,
        config: &InferenceConfig,
    ) -> Result<Tensor> {
        let dtype = positive_embeddings.dtype();

        let (lat_t, lat_h, lat_w) = self.compute_latent_dims(config);
        let mut latents = self.initialize_noise(config)?.to_dtype(dtype)?;

        let sample_shape = (1, self.vae_config.latent_channels, lat_t, lat_h, lat_w);
        self.scheduler.set_timesteps_with_shape(
            self.scheduler.num_steps(),
            Some(sample_shape),
            None,
            None,
        )?;

        let indices_grid = crate::rope::generate_indices_grid_diffusers(
            1,
            lat_t,
            lat_h,
            lat_w,
            config.frame_rate,
            &self.device,
        )?;

        let cfg_embeddings =
            self.prepare_cfg_embeddings(positive_embeddings, negative_embeddings)?;
        let num_steps = self.scheduler.num_steps();
        for step_idx in 0..num_steps {
            let timestep = self.scheduler.timesteps()[step_idx];
            let timestep_tensor = self.create_timestep_tensor(timestep, 2)?.to_dtype(dtype)?;

            let cfg_latents = self.duplicate_for_cfg(&latents)?;
            let velocity = self.require_dit()?.forward(
                &cfg_latents,
                &indices_grid,
                Some(&cfg_embeddings),
                &timestep_tensor,
                None,
                None,
                None,
                None,
            )?;

            let guided_velocity = self.apply_cfg(&velocity)?;
            latents = self.scheduler.step(&guided_velocity, &latents, step_idx)?;
        }

        let latents = self.denormalize_latents(&latents)?;
        let decode_timestep = if self.vae_config.timestep_conditioning {
            Some(0.05)
        } else {
            None
        };

        self.require_vae_decoder()?
            .decode(&latents, decode_timestep)
    }

    // =========================================================================
    // Image-to-Video Generation
    // =========================================================================

    fn image_to_tensor(
        image: &DynamicImage,
        height: usize,
        width: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let resized = image.resize_exact(width as u32, height as u32, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let mut data = vec![0f32; 3 * height * width];
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let r = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                let g = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
                let b = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;

                let idx = y * width + x;
                data[idx] = r;
                data[height * width + idx] = g;
                data[2 * height * width + idx] = b;
            }
        }

        Tensor::from_vec(data, (1, 3, 1, height, width), device)
    }

    /// Encode a conditioning image into VAE latents for i2v.
    pub fn conditioning_item_from_image(
        &self,
        image: &DynamicImage,
        config: &InferenceConfig,
        strength: f32,
    ) -> Result<ConditioningItem> {
        let encoder = self
            .vae_encoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("VAE encoder not loaded".to_string()))?;
        let (mean, std) = self.require_latent_stats()?;

        let image_tensor = Self::image_to_tensor(image, config.height, config.width, &self.device)?;
        let latents = crate::vae::vae_encode(
            encoder,
            &image_tensor,
            Some(mean),
            Some(std),
            self.vae_config.scaling_factor,
        )?;

        Ok(ConditioningItem::new(latents, 0, strength))
    }

    /// Prepare conditioning latents from conditioning items
    ///
    /// This method integrates conditioning frames into the latent tensor
    /// and creates the conditioning mask for per-token denoising.
    ///
    /// # Arguments
    /// * `init_latents` - Initial noise latents [B, C, T, H, W]
    /// * `conditioning_items` - List of conditioning items (encoded frames)
    ///
    /// # Returns
    /// Tuple of (conditioned_latents, conditioning_mask)
    /// - conditioned_latents: [B, C, T, H, W] with conditioning integrated
    /// - conditioning_mask: [B, seq] where 1.0 = conditioning token
    pub fn prepare_conditioning_latents(
        &self,
        init_latents: &Tensor,
        conditioning_items: &[ConditioningItem],
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _channels, t, h, w) = init_latents.dims5()?;

        // Get patch sizes for sequence length calculation
        let patch_size = self.dit_config.patch_size;
        let patch_size_t = self.dit_config.patch_size_t.unwrap_or(1);

        // Compute patched dimensions (matching pack_latents output)
        let t_patched = t / patch_size_t;
        let h_patched = h / patch_size;
        let w_patched = w / patch_size;
        let seq_len = t_patched * h_patched * w_patched;

        // Initialize conditioning mask to zeros (no conditioning)
        let mut mask_data = vec![0.0f32; batch * seq_len];
        let mut latents = init_latents.clone();

        for item in conditioning_items {
            let (_, _, cond_t, _cond_h, _cond_w) = item.latents.dims5()?;

            // Update latents with conditioning values for the specified frames
            if item.frame_number == 0 && cond_t <= t {
                // Create mask with conditioning strength for first cond_t frames (in patched space)
                let cond_t_patched = cond_t / patch_size_t;
                let num_cond_tokens = cond_t_patched * h_patched * w_patched;
                for b in 0..batch {
                    for i in 0..num_cond_tokens {
                        mask_data[b * seq_len + i] = item.strength;
                    }
                }

                // Replace first frames with conditioning latents
                if cond_t < t {
                    let cond_part = item.latents.i((.., .., 0..cond_t, .., ..))?;
                    let rest_part = latents.i((.., .., cond_t.., .., ..))?;
                    latents = Tensor::cat(&[&cond_part, &rest_part], 2)?;
                } else {
                    latents = item.latents.clone();
                }
            }
        }

        let conditioning_mask = Tensor::from_vec(mask_data, (batch, seq_len), &self.device)?;
        Ok((latents, conditioning_mask))
    }

    /// Mock denoising step with conditioning mask
    ///
    /// Uses per-token timesteps based on conditioning mask
    pub fn mock_denoising_step_with_mask(
        &self,
        latents: &Tensor,
        _text_emb: &Tensor,
        step_idx: usize,
        conditioning_mask: &Tensor,
        init_latents: &Tensor,
        noise_scale: f64,
    ) -> Result<Tensor> {
        let timestep = self.scheduler.timesteps()[step_idx];

        // Compute per-token timesteps from conditioning mask
        let (_b, _c, t, h, w) = latents.dims5()?;
        let patch_size = self.dit_config.patch_size;
        let patch_size_t = self.dit_config.patch_size_t.unwrap_or(1);

        // Pack latents to [B, seq, C] for per-token processing
        let packed = pack_latents(latents, patch_size, patch_size_t)?;

        // Compute per-token timesteps
        let token_timesteps = compute_token_timesteps(timestep, conditioning_mask)?;

        // Add noise to conditioning latents (for motion continuity)
        let init_packed = pack_latents(init_latents, patch_size, patch_size_t)?;
        let noised = add_noise_to_conditioning_latents(
            timestep,
            &init_packed,
            &packed,
            noise_scale,
            conditioning_mask,
        )?;

        // Mock model forward (in real implementation, this would call DiT)
        let velocity = Tensor::randn(0f32, 0.1, noised.shape(), &self.device)?;

        // Per-token Euler step
        let denoised = self
            .scheduler
            .step_per_token(&velocity, &noised, &token_timesteps)?;

        // Apply conditioning mask to preserve conditioned tokens
        let result = apply_conditioning_mask(&denoised, &packed, timestep, conditioning_mask)?;

        // Unpack back to 5D
        unpack_latents(&result, t, h, w, patch_size, patch_size_t)
    }

    /// Mock image-to-video generation
    ///
    /// Generates video conditioned on input image(s)
    ///
    /// # Arguments
    /// * `text_emb` - Text embeddings
    /// * `conditioning_items` - Conditioning items (encoded input frames)
    /// * `config` - Inference configuration
    /// * `noise_scale` - Noise scale for conditioning (default: 0.025)
    pub fn mock_generate_image_to_video(
        &self,
        text_emb: &Tensor,
        conditioning_items: &[ConditioningItem],
        config: &InferenceConfig,
        noise_scale: f64,
    ) -> Result<Tensor> {
        // Initialize noise
        let init_noise = self.initialize_noise(config)?;

        // Prepare conditioning latents and mask
        let (mut latents, conditioning_mask) =
            self.prepare_conditioning_latents(&init_noise, conditioning_items)?;

        // Store initial latents for noise injection
        let init_latents = latents.clone();

        // Denoising loop
        let num_steps = self.scheduler.num_steps();
        for step in 0..num_steps {
            latents = self.mock_denoising_step_with_mask(
                &latents,
                text_emb,
                step,
                &conditioning_mask,
                &init_latents,
                noise_scale,
            )?;
        }

        // Decode to video
        self.mock_vae_decode(&latents, config)
    }

    /// Generate image-to-video (public API)
    ///
    /// # Arguments
    /// * `text_embeddings` - Text embeddings
    /// * `conditioning_items` - Encoded conditioning frames
    /// * `config` - Inference configuration
    pub fn generate_image_to_video(
        &mut self,
        text_embeddings: &Tensor,
        conditioning_items: &[ConditioningItem],
        config: &InferenceConfig,
    ) -> Result<Tensor> {
        let dtype = text_embeddings.dtype();

        let (lat_t, lat_h, lat_w) = self.compute_latent_dims(config);
        let init_noise = self.initialize_noise(config)?.to_dtype(dtype)?;

        let (mut latents, conditioning_mask) =
            self.prepare_conditioning_latents(&init_noise, conditioning_items)?;
        let init_latents = latents.clone();

        let sample_shape = (1, self.vae_config.latent_channels, lat_t, lat_h, lat_w);
        self.scheduler.set_timesteps_with_shape(
            self.scheduler.num_steps(),
            Some(sample_shape),
            None,
            None,
        )?;

        let indices_grid = crate::rope::generate_indices_grid_diffusers(
            1,
            lat_t,
            lat_h,
            lat_w,
            config.frame_rate,
            &self.device,
        )?;

        let patch_size = self.dit_config.patch_size;
        let patch_size_t = self.dit_config.patch_size_t.unwrap_or(1);
        let noise_scale = 0.025;
        let num_steps = self.scheduler.num_steps();

        for step_idx in 0..num_steps {
            let timestep = self.scheduler.timesteps()[step_idx];

            let packed = pack_latents(&latents, patch_size, patch_size_t)?;
            let token_timesteps = compute_token_timesteps(timestep, &conditioning_mask)?;
            let init_packed = pack_latents(&init_latents, patch_size, patch_size_t)?;
            let noised_packed = add_noise_to_conditioning_latents(
                timestep,
                &init_packed,
                &packed,
                noise_scale,
                &conditioning_mask,
            )?;

            let noised = unpack_latents(
                &noised_packed,
                lat_t,
                lat_h,
                lat_w,
                patch_size,
                patch_size_t,
            )?;
            let timestep_tensor = Tensor::new(&[timestep as f32], &self.device)?.to_dtype(dtype)?;
            let velocity = self.require_dit()?.forward(
                &noised,
                &indices_grid,
                Some(text_embeddings),
                &timestep_tensor,
                None,
                None,
                None,
                None,
            )?;
            let velocity_packed = pack_latents(&velocity, patch_size, patch_size_t)?;

            let denoised_packed = self.scheduler.step_per_token(
                &velocity_packed,
                &noised_packed,
                &token_timesteps,
            )?;
            let result_packed =
                apply_conditioning_mask(&denoised_packed, &packed, timestep, &conditioning_mask)?;

            latents = unpack_latents(
                &result_packed,
                lat_t,
                lat_h,
                lat_w,
                patch_size,
                patch_size_t,
            )?;
        }

        let latents = self.denormalize_latents(&latents)?;
        let decode_timestep = if self.vae_config.timestep_conditioning {
            Some(0.05)
        } else {
            None
        };

        self.require_vae_decoder()?
            .decode(&latents, decode_timestep)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn create_test_config() -> PipelineConfig {
        PipelineConfig {
            dit: DitConfig::default(),
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig::default(),
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let device = Device::Cpu;
        let config = create_test_config();

        let pipeline = TextToVideoPipeline::new(device, config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_noise_initialization() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let pipeline = TextToVideoPipeline::new(device, config)?;

        let inference_config = InferenceConfig::new(17, 512, 768, 42)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let noise = pipeline.initialize_noise(&inference_config)?;

        // Check shape (B, C_latent, T_latent, H_latent, W_latent)
        assert_eq!(noise.dim(0)?, 1);
        assert_eq!(noise.dim(1)?, 128); // latent_channels

        Ok(())
    }

    #[test]
    fn test_compute_latent_dims() {
        let device = Device::Cpu;
        let config = create_test_config();
        let pipeline = TextToVideoPipeline::new(device, config).unwrap();

        let inference_config = InferenceConfig::new(17, 512, 768, 42).unwrap();
        let (lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference_config);

        // With default config: temporal_downsample=8, spatial_downsample=32
        assert_eq!(lat_t, 3); // (17 - 1) / 8 + 1 = 3
        assert_eq!(lat_h, 16); // 512 / 32 = 16
        assert_eq!(lat_w, 24); // 768 / 32 = 24
    }

    #[test]
    fn test_scheduler_accessors() {
        let device = Device::Cpu;
        let config = PipelineConfig {
            scheduler: SchedulerConfig {
                num_inference_steps: 25,
                guidance_scale: 5.0,
                ..SchedulerConfig::default()
            },
            ..create_test_config()
        };
        let pipeline = TextToVideoPipeline::new(device, config).unwrap();

        assert_eq!(pipeline.scheduler().num_steps(), 25);
        assert_eq!(pipeline.scheduler().guidance_scale(), 5.0);
    }

    #[test]
    fn test_cfg_embeddings_preparation() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

        let pos_emb = Tensor::ones((1, 10, 100), DType::F32, &device)?;
        let neg_emb = Tensor::zeros((1, 10, 100), DType::F32, &device)?;

        let cfg_emb = pipeline.prepare_cfg_embeddings(&pos_emb, &neg_emb)?;

        // Should concatenate along batch dimension
        assert_eq!(cfg_emb.dims(), &[2, 10, 100]);

        Ok(())
    }

    #[test]
    fn test_mock_generation() -> Result<()> {
        let device = Device::Cpu;
        let config = PipelineConfig {
            dit: DitConfig::default(),
            vae: VaeConfig::default(),
            scheduler: SchedulerConfig {
                num_inference_steps: 3,
                ..SchedulerConfig::default()
            },
        };
        let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

        let inference_config = InferenceConfig::new(17, 512, 768, 42)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 4096), &device)?;

        let video = pipeline.mock_generate(&text_emb, &inference_config)?;

        // Video shape: (B, 3, T, H, W)
        assert_eq!(video.dim(0)?, 1);
        assert_eq!(video.dim(1)?, 3);
        assert_eq!(video.dim(2)?, 17);
        assert_eq!(video.dim(3)?, 512);
        assert_eq!(video.dim(4)?, 768);

        Ok(())
    }
}
