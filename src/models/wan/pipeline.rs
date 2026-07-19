//! Wan 2.1 text-to-video pipeline (Diffusers `WanPipeline` parity, T2V only).

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use crate::engine::model::ModelCapabilities;
use crate::engine::{ensure_wan_cuda_requirements, wan_patch_token_count};
use crate::profile_zone;
use crate::utils::deterministic_rng::Pcg32;

use super::configs::WanFullConfig;
use super::loader::{
    WanLayout, detect_wan_layout, load_wan_config, wan_scheduler_config_path, wan_tokenizer_path,
};
use super::prompt_clean::prompt_clean;
use super::prompt_encode::encode_prompt;
use super::scheduler::{WanScheduler, WanSchedulerProfile};
use super::text_encoder::Umt5TextEncoder;
use super::tokenizer::{WAN_DEFAULT_MAX_SEQ_LEN, WanTokenizer};
use super::transformer::WanTransformer3DModel;
use super::vae::AutoencoderKLWan;

/// User-facing Wan generation parameters.
#[derive(Debug, Clone)]
pub struct WanGenerateRequest {
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub max_sequence_length: usize,
    pub initial_latents: Option<Tensor>,
    pub prompt_embeds: Option<Tensor>,
    pub negative_prompt_embeds: Option<Tensor>,
}

impl WanGenerateRequest {
    pub fn text_to_video(
        prompt: impl Into<String>,
        height: usize,
        width: usize,
        num_frames: usize,
        num_inference_steps: usize,
    ) -> Self {
        Self {
            prompt: Some(prompt.into()),
            negative_prompt: None,
            height,
            width,
            num_frames,
            num_inference_steps,
            guidance_scale: 5.0,
            seed: None,
            max_sequence_length: WAN_DEFAULT_MAX_SEQ_LEN,
            initial_latents: None,
            prompt_embeds: None,
            negative_prompt_embeds: None,
        }
    }
}

/// Decoded Wan video output (`[B, C, T, H, W]` in `[-1, 1]`).
#[derive(Debug, Clone)]
pub struct WanPipelineOutput {
    pub frames: Tensor,
}

/// Latent tensor shape for Wan T2V (Diffusers `prepare_latents`).
pub fn latent_shape_from_config(
    config: &WanFullConfig,
    batch: usize,
    height: usize,
    width: usize,
    num_frames: usize,
) -> Vec<usize> {
    let temporal = config.temporal_compression();
    let spatial = config.spatial_compression();
    let num_latent_frames = num_frames.saturating_sub(1) / temporal + 1;
    vec![
        batch,
        config.latent_channels(),
        num_latent_frames,
        height / spatial,
        width / spatial,
    ]
}

/// Transformer + VAE + scheduler stack (no text encoder — use precomputed embeds).
pub struct WanDenoiseStack {
    config: WanFullConfig,
    transformer: Option<WanTransformer3DModel>,
    vae: Option<AutoencoderKLWan>,
    scheduler: WanScheduler,
    scheduler_profile: WanSchedulerProfile,
    device: Device,
    vae_device: Device,
    transformer_dtype: DType,
    vae_dtype: DType,
    layout: WanLayout,
    compute_device: Device,
    vae_tiling: bool,
}

impl WanDenoiseStack {
    pub fn load(
        root: impl AsRef<Path>,
        device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_vae_device(
            root,
            device,
            device,
            device,
            transformer_dtype,
            vae_dtype,
            false,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with_vae_device(
        root: impl AsRef<Path>,
        transformer_device: &Device,
        vae_device: &Device,
        compute_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
        defer_transformer: bool,
        vae_tiling: bool,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_scheduler_profile(
            root,
            transformer_device,
            vae_device,
            compute_device,
            transformer_dtype,
            vae_dtype,
            defer_transformer,
            vae_tiling,
            WanSchedulerProfile::Diffusers,
            None,
        )
    }

    /// Load the denoising stack with an explicit scheduler compatibility profile.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_scheduler_profile(
        root: impl AsRef<Path>,
        transformer_device: &Device,
        vae_device: &Device,
        compute_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
        defer_transformer: bool,
        vae_tiling: bool,
        scheduler_profile: WanSchedulerProfile,
        scheduler_shift: Option<f32>,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        let root = root.as_ref().to_path_buf();
        let layout = detect_wan_layout(&root)?;
        let config = load_wan_config(&root)?;
        let transformer = if defer_transformer {
            None
        } else {
            Some(WanTransformer3DModel::load_with_layout(
                &layout,
                transformer_device,
                transformer_dtype,
                &config.transformer,
            )?)
        };
        let mut vae =
            AutoencoderKLWan::load_with_layout(&layout, &config.vae, vae_device, vae_dtype)?;
        if vae_tiling {
            vae.enable_tiling();
        }
        let scheduler_config = match wan_scheduler_config_path(&layout) {
            Some(path) => crate::models::ltx_video::loader::load_model_config(&path)?,
            None => config.scheduler.clone(),
        };
        let scheduler =
            WanScheduler::from_profile(scheduler_profile, scheduler_config, scheduler_shift)
                .map_err(crate::models::ltx_video::loader::LoaderError::Candle)?;

        Ok(Self {
            config,
            transformer,
            vae: Some(vae),
            scheduler,
            scheduler_profile,
            device: if defer_transformer {
                compute_device.clone()
            } else {
                transformer_device.clone()
            },
            vae_device: vae_device.clone(),
            transformer_dtype,
            vae_dtype,
            layout,
            compute_device: compute_device.clone(),
            vae_tiling,
        })
    }

    /// Move transformer weights to `compute_device` (Diffusers sequential offload after TE encode).
    pub fn ensure_transformer_on_compute(&mut self) -> Result<()> {
        if self.transformer.is_some() && self.device.is_cuda() {
            return Ok(());
        }
        self.transformer = Some(
            WanTransformer3DModel::load_with_layout(
                &self.layout,
                &self.compute_device,
                self.transformer_dtype,
                &self.config.transformer,
            )
            .map_err(candle_core::Error::wrap)?,
        );
        self.device = self.compute_device.clone();
        Ok(())
    }

    /// Load VAE on `compute_device` for decode (Diffusers sequential offload after denoise).
    pub fn ensure_vae_on_compute(&mut self) -> Result<()> {
        if self.vae_device.is_cuda() && self.vae.is_some() {
            return Ok(());
        }
        self.vae = Some(
            AutoencoderKLWan::load_with_layout(
                &self.layout,
                &self.config.vae,
                &self.compute_device,
                self.vae_dtype,
            )
            .map_err(candle_core::Error::wrap)?,
        );
        if let Some(vae) = self.vae.as_mut()
            && self.vae_tiling
        {
            vae.enable_tiling();
        }
        self.vae_device = self.compute_device.clone();
        Ok(())
    }

    pub fn config(&self) -> &WanFullConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Scheduler profile used by this stack.
    pub fn scheduler_profile(&self) -> WanSchedulerProfile {
        self.scheduler_profile
    }

    pub fn transformer(&self) -> Result<&WanTransformer3DModel> {
        self.transformer
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("transformer not loaded".into()))
    }

    pub fn vae_mut(&mut self) -> Option<&mut AutoencoderKLWan> {
        self.vae.as_mut()
    }

    pub fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        num_frames: usize,
        seed: Option<u64>,
        latents: Option<Tensor>,
    ) -> Result<Tensor> {
        if let Some(latents) = latents {
            return latents.to_device(&self.device);
        }
        let shape = latent_shape_from_config(&self.config, batch_size, height, width, num_frames);
        let mut rng = Pcg32::new(seed.unwrap_or(0), 0);
        rng.randn(shape, &self.device)
    }

    /// Denoise with precomputed prompt embeddings, then VAE-decode to pixels.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_and_decode(
        &mut self,
        height: usize,
        width: usize,
        num_frames: usize,
        num_inference_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        initial_latents: Option<Tensor>,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Option<Tensor>,
    ) -> Result<WanPipelineOutput> {
        let latents = self.denoise(
            height,
            width,
            num_frames,
            num_inference_steps,
            guidance_scale,
            seed,
            initial_latents,
            prompt_embeds,
            negative_prompt_embeds,
        )?;

        // Drop transformer before VAE decode so 12 GB GPUs fit Wan + VAE sequentially.
        if self.compute_device.is_cuda() {
            self.transformer = None;
            self.compute_device.synchronize()?;
            crate::profiling::vram::log_vram("post_denoise_transformer_drop");
        }

        profile_zone!("wan_vae_decode");
        if !self.vae_device.is_cuda() && self.compute_device.is_cuda() {
            self.ensure_vae_on_compute()?;
            crate::profiling::vram::log_vram("post_vae_load");
        }
        let vae = self
            .vae
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("vae not loaded".into()))?;
        let latents_for_vae = latents
            .to_dtype(self.vae_dtype)?
            .to_device(&self.vae_device)?;
        let frames = vae.decode(&latents_for_vae)?;
        let frames = frames.to_device(&self.device)?;
        Ok(WanPipelineOutput { frames })
    }

    /// Denoise only (no VAE decode) — for profiling and latent export.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &mut self,
        height: usize,
        width: usize,
        num_frames: usize,
        num_inference_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        initial_latents: Option<Tensor>,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Option<Tensor>,
    ) -> Result<Tensor> {
        if height == 0 || width == 0 || !height.is_multiple_of(16) || !width.is_multiple_of(16) {
            candle_core::bail!(
                "Wan denoise requires positive height/width divisible by 16, got {}x{}",
                height,
                width
            );
        }
        if num_frames == 0 {
            candle_core::bail!("Wan denoise requires num_frames >= 1");
        }
        if num_inference_steps == 0 {
            candle_core::bail!("Wan denoise requires num_inference_steps >= 1");
        }
        let batch_size = prompt_embeds.dim(0)?;
        let do_cfg = guidance_scale > 1.0;
        let num_frames = normalize_num_frames(num_frames, self.config.temporal_compression());

        let tokens = wan_patch_token_count(height, width, num_frames);
        ensure_wan_cuda_requirements(&self.device, tokens)?;

        let prompt_embeds = prompt_embeds.to_dtype(self.transformer_dtype)?;
        let negative_embeds = negative_prompt_embeds
            .map(|t| t.to_dtype(self.transformer_dtype))
            .transpose()?;

        if do_cfg && negative_embeds.is_none() {
            candle_core::bail!("negative_prompt_embeds required when guidance_scale > 1");
        }

        self.scheduler.set_timesteps(num_inference_steps)?;
        self.scheduler.set_begin_index(0);
        let timesteps = self.scheduler.timesteps_f32();
        let floating_timesteps = self.scheduler_profile == WanSchedulerProfile::FlowMatchEuler;

        let mut latents =
            self.prepare_latents(batch_size, height, width, num_frames, seed, initial_latents)?;

        for (step_i, t) in timesteps.iter().copied().enumerate() {
            let _ = step_i;
            profile_zone!("wan_denoise_step", step = step_i, total = timesteps.len());
            // ponytail: keep `latent_model_input` and `noise_pred` in the
            // transformer dtype across the loop. Previously we converted
            // every step (to F16, then F32 → scheduler → F16 → scheduler);
            // that's two extra GPU copies × 50 steps = 100 avoidable copies.
            let latent_model_input = if latents.dtype() == self.transformer_dtype {
                latents.clone()
            } else {
                latents.to_dtype(self.transformer_dtype)?
            };
            let timestep = if floating_timesteps {
                Tensor::full(t, batch_size, &self.device)?
            } else {
                Tensor::full(t as i64, batch_size, &self.device)?
            };

            profile_zone!("wan_transformer_forward_cond");
            let transformer = self
                .transformer
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("transformer not loaded".into()))?;
            let noise_pred = transformer.forward(&latent_model_input, &timestep, &prompt_embeds)?;

            let noise_pred = if do_cfg {
                let neg = negative_embeds.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg(
                        "negative embeddings are required for classifier-free guidance".into(),
                    )
                })?;
                profile_zone!("wan_transformer_forward_uncond");
                let noise_uncond = transformer.forward(&latent_model_input, &timestep, neg)?;
                WanPipeline::apply_classifier_free_guidance(
                    &noise_pred,
                    &noise_uncond,
                    guidance_scale,
                )?
            } else {
                noise_pred
            };
            // Scheduler works in transformer_dtype now (scalar `affine` keeps
            // precision); skip the F32 round-trip per step.

            latents = self
                .scheduler
                .step_f32(&noise_pred, t, &latents)?
                .prev_sample;
        }

        Ok(latents)
    }
}

/// Loaded Wan T2V pipeline (single transformer stage).
pub struct WanPipeline {
    config: WanFullConfig,
    tokenizer: WanTokenizer,
    text_encoder: Option<Umt5TextEncoder>,
    stack: WanDenoiseStack,
    layout: WanLayout,
    compute_device: Device,
    text_encoder_storage: Device,
    transformer_dtype: DType,
    sequential_gpu: bool,
}

impl WanPipeline {
    pub fn load(
        root: impl AsRef<Path>,
        device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_devices(
            root,
            device,
            device,
            device,
            transformer_dtype,
            vae_dtype,
            false,
            false,
        )
    }

    /// Load pipeline; `text_encoder_device` is often `Cpu` on 12 GB GPUs (UMT5-XXL is large).
    pub fn load_with_text_encoder_device(
        root: impl AsRef<Path>,
        device: &Device,
        text_encoder_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_devices(
            root,
            device,
            text_encoder_device,
            device,
            transformer_dtype,
            vae_dtype,
            false,
            false,
        )
    }

    /// Load pipeline with per-component devices (e.g. VAE on CPU to save VRAM).
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_devices(
        root: impl AsRef<Path>,
        compute_device: &Device,
        text_encoder_device: &Device,
        vae_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
        sequential_gpu: bool,
        vae_tiling: bool,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_devices_and_scheduler(
            root,
            compute_device,
            text_encoder_device,
            vae_device,
            transformer_dtype,
            vae_dtype,
            sequential_gpu,
            vae_tiling,
            WanSchedulerProfile::Diffusers,
            None,
        )
    }

    /// Load a pipeline with an explicit Wan scheduler profile and optional shift.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_devices_and_scheduler(
        root: impl AsRef<Path>,
        compute_device: &Device,
        text_encoder_device: &Device,
        vae_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
        sequential_gpu: bool,
        vae_tiling: bool,
        scheduler_profile: WanSchedulerProfile,
        scheduler_shift: Option<f32>,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        let layout = detect_wan_layout(&root)?;
        let config = load_wan_config(&root)?;

        let tokenizer = WanTokenizer::from_pretrained(wan_tokenizer_path(&layout))?;

        let te_device = if sequential_gpu && matches!(&layout, WanLayout::Consolidated(_)) {
            compute_device.clone()
        } else if sequential_gpu {
            // Safetensors UMT5-XXL F16 (~9 GB) OOMs alone on 12 GB; encode on CPU then swap to transformer.
            Device::Cpu
        } else {
            text_encoder_device.clone()
        };
        let te_dtype = if te_device.is_cuda() || sequential_gpu {
            transformer_dtype
        } else {
            DType::F32
        };
        let text_encoder = if sequential_gpu {
            None
        } else {
            Some(Umt5TextEncoder::load_for_layout(
                &layout, &te_device, te_dtype,
            )?)
        };

        let transformer_device = if sequential_gpu {
            Device::Cpu
        } else {
            compute_device.clone()
        };
        let stack = WanDenoiseStack::load_with_scheduler_profile(
            &root,
            &transformer_device,
            vae_device,
            compute_device,
            transformer_dtype,
            vae_dtype,
            sequential_gpu,
            vae_tiling,
            scheduler_profile,
            scheduler_shift,
        )?;

        Ok(Self {
            config,
            tokenizer,
            text_encoder,
            stack,
            layout,
            compute_device: compute_device.clone(),
            text_encoder_storage: text_encoder_device.clone(),
            transformer_dtype,
            sequential_gpu,
        })
    }

    fn release_text_encoder_after_encode(&mut self) {
        if self.sequential_gpu
            || self
                .text_encoder
                .as_ref()
                .is_some_and(|te| te.device().is_cuda())
        {
            self.text_encoder = None;
        }
    }

    fn reload_text_encoder_if_needed(
        &mut self,
    ) -> std::result::Result<(), crate::models::ltx_video::loader::LoaderError> {
        if self.text_encoder.is_none() {
            let te_device =
                if self.sequential_gpu && matches!(&self.layout, WanLayout::Consolidated(_)) {
                    self.compute_device.clone()
                } else if self.sequential_gpu {
                    Device::Cpu
                } else {
                    self.text_encoder_storage.clone()
                };
            let te_dtype = if te_device.is_cuda() || self.sequential_gpu {
                self.transformer_dtype
            } else {
                DType::F32
            };
            self.text_encoder = Some(Umt5TextEncoder::load_for_layout(
                &self.layout,
                &te_device,
                te_dtype,
            )?);
        }
        Ok(())
    }

    pub fn denoise_stack(&self) -> &WanDenoiseStack {
        &self.stack
    }

    pub fn denoise_stack_mut(&mut self) -> &mut WanDenoiseStack {
        &mut self.stack
    }

    pub fn config(&self) -> &WanFullConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        self.stack.device()
    }

    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::wan21_t2v_13b()
    }

    pub fn do_classifier_free_guidance(&self, guidance_scale: f32) -> bool {
        guidance_scale > 1.0
    }

    pub fn round_dimensions(&self, height: usize, width: usize) -> (usize, usize) {
        let patch = &self.config.transformer.patch_size;
        let h_mult = self.config.spatial_compression() * patch[1];
        let w_mult = self.config.spatial_compression() * patch[2];
        let height = height / h_mult * h_mult;
        let width = width / w_mult * w_mult;
        (height.max(h_mult), width.max(w_mult))
    }

    pub fn num_latent_frames(&self, num_frames: usize) -> usize {
        num_frames.saturating_sub(1) / self.config.temporal_compression() + 1
    }

    pub fn latent_spatial_size(&self, pixel_size: usize) -> usize {
        pixel_size / self.config.spatial_compression()
    }

    pub fn latent_shape(
        &self,
        batch: usize,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> Vec<usize> {
        latent_shape_from_config(&self.config, batch, height, width, num_frames)
    }

    pub fn check_inputs(&self, req: &WanGenerateRequest) -> Result<()> {
        Self::validate_wan_request(req)
    }

    /// Validate user-controlled generation parameters before any shape
    /// arithmetic or model execution. Keeping this separate makes the
    /// invariants reusable by adapters and easy to test without loading
    /// multi-gigabyte weights.
    fn validate_wan_request(req: &WanGenerateRequest) -> Result<()> {
        if req.height == 0 || req.width == 0 {
            candle_core::bail!(
                "height and width must be positive, got {}x{}",
                req.height,
                req.width
            );
        }
        if !req.height.is_multiple_of(16) || !req.width.is_multiple_of(16) {
            candle_core::bail!(
                "height and width must be divisible by 16, got {}x{}",
                req.height,
                req.width
            );
        }
        if req.num_frames == 0 {
            candle_core::bail!("num_frames must be at least 1, got 0");
        }
        if req.num_inference_steps == 0 {
            candle_core::bail!("num_inference_steps must be at least 1, got 0");
        }
        if !req.guidance_scale.is_finite() || req.guidance_scale < 0.0 {
            candle_core::bail!(
                "guidance_scale must be finite and non-negative, got {}",
                req.guidance_scale
            );
        }
        if req.max_sequence_length == 0 {
            candle_core::bail!("max_sequence_length must be at least 1");
        }

        let has_prompt = req.prompt.is_some();
        let has_embeds = req.prompt_embeds.is_some();
        if has_prompt == has_embeds {
            if has_prompt {
                candle_core::bail!("provide either prompt or prompt_embeds, not both");
            } else {
                candle_core::bail!("provide either prompt or prompt_embeds");
            }
        }

        if req.negative_prompt.is_some() && req.negative_prompt_embeds.is_some() {
            candle_core::bail!(
                "provide either negative_prompt or negative_prompt_embeds, not both"
            );
        }

        Ok(())
    }

    /// Gaussian latent init matching Diffusers `prepare_latents` (float32 noise).
    pub fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        num_frames: usize,
        seed: Option<u64>,
        latents: Option<Tensor>,
    ) -> Result<Tensor> {
        self.stack
            .prepare_latents(batch_size, height, width, num_frames, seed, latents)
    }

    /// Classifier-free guidance on flow/noise predictions.
    pub fn apply_classifier_free_guidance(
        noise_pred: &Tensor,
        noise_uncond: &Tensor,
        guidance_scale: f32,
    ) -> Result<Tensor> {
        let diff = noise_pred.sub(noise_uncond)?;
        noise_uncond.add(&diff.affine(guidance_scale as f64, 0.0)?)
    }

    fn encode_prompts(
        &mut self,
        req: &WanGenerateRequest,
        do_cfg: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if let Some(embeds) = &req.prompt_embeds {
            let neg = req.negative_prompt_embeds.clone();
            return Ok((embeds.clone(), neg));
        }

        let prompt = req.prompt.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("prompt is required when embeddings are absent".into())
        })?;
        let cleaned = prompt_clean(prompt);
        let prompts = vec![cleaned];
        let negative = req.negative_prompt.as_ref().map(|n| vec![prompt_clean(n)]);

        let te_device = self
            .text_encoder
            .as_ref()
            .map(|te| te.device().clone())
            .unwrap_or_else(|| self.text_encoder_storage.clone());
        let infer_device = self.compute_device.clone();
        let text_encoder = self
            .text_encoder
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("text encoder not loaded".into()))?;
        let (prompt_embeds, negative_embeds) = encode_prompt(
            &self.tokenizer,
            text_encoder,
            &prompts,
            negative.as_deref(),
            do_cfg,
            req.max_sequence_length,
            &te_device,
        )?;
        let prompt_embeds = prompt_embeds.to_device(&infer_device)?;
        let negative_embeds = match negative_embeds {
            Some(n) => Some(n.to_device(&infer_device)?),
            None => None,
        };
        Ok((prompt_embeds, negative_embeds))
    }

    /// Run denoising and VAE decode; returns pixel tensor `[B, C, T, H, W]`.
    pub fn generate(&mut self, req: WanGenerateRequest) -> Result<WanPipelineOutput> {
        use std::time::Instant;

        self.check_inputs(&req)?;

        let (height, width) = self.round_dimensions(req.height, req.width);
        let do_cfg = self.do_classifier_free_guidance(req.guidance_scale);

        let t0 = Instant::now();
        let (prompt_embeds, negative_embeds) = if let Some(embeds) = &req.prompt_embeds {
            (embeds.clone(), req.negative_prompt_embeds.clone())
        } else {
            self.reload_text_encoder_if_needed()
                .map_err(candle_core::Error::wrap)?;
            let pair = self.encode_prompts(&req, do_cfg)?;
            self.release_text_encoder_after_encode();
            pair
        };
        eprintln!("encode_prompt: {:.2}s", t0.elapsed().as_secs_f64());

        if self.sequential_gpu {
            let t_load = Instant::now();
            self.stack.ensure_transformer_on_compute()?;
            eprintln!("load_transformer: {:.2}s", t_load.elapsed().as_secs_f64());
        }

        let t1 = Instant::now();
        let out = self.stack.denoise_and_decode(
            height,
            width,
            req.num_frames,
            req.num_inference_steps,
            req.guidance_scale,
            req.seed,
            req.initial_latents,
            prompt_embeds,
            negative_embeds,
        )?;
        eprintln!("denoise+decode: {:.2}s", t1.elapsed().as_secs_f64());
        Ok(out)
    }
}

fn normalize_num_frames(num_frames: usize, temporal_compression: usize) -> usize {
    let adjusted = if !(num_frames.saturating_sub(1)).is_multiple_of(temporal_compression) {
        num_frames / temporal_compression * temporal_compression + 1
    } else {
        num_frames
    };
    adjusted.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> WanFullConfig {
        use super::super::configs::*;
        WanFullConfig {
            variant: WanVariant::Wan21T2v13B,
            model_index: WanModelIndex {
                class_name: "WanPipeline".to_string(),
                diffusers_version: Some("0.33.0.dev0".to_string()),
                scheduler: ("scheduler".into(), "UniPCMultistepScheduler".into()),
                text_encoder: ("text_encoder".into(), "UMT5EncoderModel".into()),
                tokenizer: ("tokenizer".into(), "T5Tokenizer".into()),
                transformer: ("transformer".into(), "WanTransformer3DModel".into()),
                vae: ("vae".into(), "AutoencoderKLWan".into()),
            },
            transformer: WanTransformerConfig {
                attention_head_dim: 128,
                cross_attn_norm: true,
                eps: 1e-6,
                ffn_dim: 8960,
                freq_dim: 256,
                in_channels: 16,
                num_attention_heads: 12,
                num_layers: 30,
                out_channels: 16,
                patch_size: [1, 2, 2],
                qk_norm: "rms_norm_across_heads".to_string(),
                rope_max_seq_len: 1024,
                text_dim: 4096,
                image_dim: None,
                added_kv_proj_dim: None,
            },
            vae: WanVaeConfig {
                base_dim: 96,
                dim_mult: vec![1, 2, 4, 4],
                dropout: 0.0,
                latents_mean: vec![0.0; 16],
                latents_std: vec![1.0; 16],
                num_res_blocks: 2,
                temporal_downsample: vec![false, true, true],
                z_dim: 16,
                attn_scales: vec![],
            },
            scheduler: WanSchedulerConfig {
                num_train_timesteps: 1000,
                flow_shift: 3.0,
                prediction_type: "flow_prediction".to_string(),
                use_flow_sigmas: true,
                solver_order: 2,
                solver_type: "bh2".to_string(),
                beta_start: 0.0001,
                beta_end: 0.02,
                beta_schedule: "linear".to_string(),
                timestep_spacing: "linspace".to_string(),
                predict_x0: true,
                lower_order_final: true,
                final_sigmas_type: "zero".to_string(),
                disable_corrector: vec![],
            },
            text_encoder: WanTextEncoderConfig {
                d_model: 4096,
                d_ff: 10240,
                d_kv: 64,
                num_heads: 64,
                num_layers: 24,
                vocab_size: 256_384,
                model_type: "umt5".to_string(),
                feed_forward_proj: "gated-gelu".to_string(),
                layer_norm_epsilon: 1e-6,
                dropout_rate: 0.1,
            },
        }
    }

    #[test]
    fn latent_shape_matches_diffusers_formula() {
        let config = dummy_config();
        let temporal = config.temporal_compression();
        let spatial = config.spatial_compression();
        assert_eq!((81 - 1) / temporal + 1, 21);
        assert_eq!(480 / spatial, 60);
        assert_eq!(832 / spatial, 104);
        assert_eq!(
            latent_shape_from_config(&config, 1, 32, 32, 0),
            vec![1, 16, 1, 4, 4]
        );
    }

    #[test]
    fn normalize_num_frames_rounds_to_temporal_step() {
        assert_eq!(normalize_num_frames(82, 4), 81);
        assert_eq!(normalize_num_frames(81, 4), 81);
        assert_eq!(normalize_num_frames(1, 4), 1);
        assert_eq!(normalize_num_frames(0, 4), 1);
    }

    #[test]
    fn request_validation_rejects_empty_video_and_zero_steps() {
        let mut request = WanGenerateRequest::text_to_video("cat", 32, 32, 1, 1);
        request.num_frames = 0;
        let error = WanPipeline::validate_wan_request(&request).expect_err("zero frames must fail");
        assert!(error.to_string().contains("num_frames"));

        request.num_frames = 1;
        request.num_inference_steps = 0;
        let error = WanPipeline::validate_wan_request(&request).expect_err("zero steps must fail");
        assert!(error.to_string().contains("num_inference_steps"));
    }

    #[test]
    fn cfg_formula_matches_reference() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).expect("a");
        let b = Tensor::new(&[0.5f32, 1.0, 1.5], &device).expect("b");
        let out = WanPipeline::apply_classifier_free_guidance(&a, &b, 5.0).expect("cfg");
        let v = out.to_vec1::<f32>().expect("vec");
        assert!((v[0] - 3.0).abs() < 1e-5);
        assert!((v[1] - 6.0).abs() < 1e-5);
        assert!((v[2] - 9.0).abs() < 1e-5);
    }
}
