//! Wan 2.1 text-to-video pipeline (Diffusers `WanPipeline` parity, T2V only).

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use crate::engine::model::ModelCapabilities;
use crate::utils::debug_log::{debug_log, tensor_stats};
use crate::utils::deterministic_rng::Pcg32;

use super::configs::WanFullConfig;
use super::loader::{load_wan_config, WanComponentPaths};
use super::prompt_clean::prompt_clean;
use super::prompt_encode::encode_prompt;
use super::scheduler::UniPcMultistepScheduler;
use super::text_encoder::Umt5TextEncoder;
use super::tokenizer::{WanTokenizer, WAN_DEFAULT_MAX_SEQ_LEN};
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
    let num_latent_frames = (num_frames - 1) / temporal + 1;
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
    transformer: WanTransformer3DModel,
    vae: AutoencoderKLWan,
    scheduler: UniPcMultistepScheduler,
    device: Device,
    vae_device: Device,
    transformer_dtype: DType,
    vae_dtype: DType,
}

impl WanDenoiseStack {
    pub fn load(
        root: impl AsRef<Path>,
        device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_vae_device(root, device, device, transformer_dtype, vae_dtype)
    }

    pub fn load_with_vae_device(
        root: impl AsRef<Path>,
        device: &Device,
        vae_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        let paths = WanComponentPaths::from_root(&root);
        paths.validate_exists()?;
        let config = load_wan_config(&root)?;
        let transformer =
            WanTransformer3DModel::load(&paths.root.join("transformer"), device, transformer_dtype)?;
        let vae = AutoencoderKLWan::load(&paths.root.join("vae"), vae_device, vae_dtype)?;
        let scheduler = UniPcMultistepScheduler::from_config_path(&paths.scheduler_config)?;

        Ok(Self {
            config,
            transformer,
            vae,
            scheduler,
            device: device.clone(),
            vae_device: vae_device.clone(),
            transformer_dtype,
            vae_dtype,
        })
    }

    pub fn config(&self) -> &WanFullConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
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
        let batch_size = prompt_embeds.dim(0)?;
        let do_cfg = guidance_scale > 1.0;
        let num_frames = normalize_num_frames(num_frames, self.config.temporal_compression());

        let prompt_embeds = prompt_embeds.to_dtype(self.transformer_dtype)?;
        let negative_embeds = negative_prompt_embeds
            .map(|t| t.to_dtype(self.transformer_dtype))
            .transpose()?;

        if do_cfg && negative_embeds.is_none() {
            candle_core::bail!("negative_prompt_embeds required when guidance_scale > 1");
        }

        self.scheduler.set_timesteps(num_inference_steps)?;
        self.scheduler.set_begin_index(0);
        let timesteps: Vec<i64> = self.scheduler.timesteps().to_vec();

        let mut latents = self.prepare_latents(
            batch_size,
            height,
            width,
            num_frames,
            seed,
            initial_latents,
        )?;
        // #region agent log
        if let Ok(s) = tensor_stats("latents_init", &latents) {
            debug_log("D", "pipeline.rs:init", "initial latents", s);
        }
        // #endregion

        for (step_i, t) in timesteps.iter().copied().enumerate() {
            let latent_model_input = latents.to_dtype(self.transformer_dtype)?;
            let timestep = Tensor::full(t, batch_size, &self.device)?;

            let noise_pred =
                self.transformer
                    .forward(&latent_model_input, &timestep, &prompt_embeds)?;

            let noise_pred = if do_cfg {
                let neg = negative_embeds.as_ref().expect("negative embeds");
                let noise_uncond = self.transformer.forward(
                    &latent_model_input,
                    &timestep,
                    neg,
                )?;
                WanPipeline::apply_classifier_free_guidance(
                    &noise_pred,
                    &noise_uncond,
                    guidance_scale,
                )?
            } else {
                noise_pred
            };
            let noise_pred = noise_pred.to_dtype(DType::F32)?;

            latents = self
                .scheduler
                .step(&noise_pred, t, &latents)?
                .prev_sample;

            // #region agent log
            if step_i == 0 || step_i + 1 == timesteps.len() {
                if let Ok(s) = tensor_stats(&format!("latents_step_{step_i}"), &latents) {
                    debug_log("C", "pipeline.rs:step", "latents after scheduler", s);
                }
            }
            // #endregion
        }

        // #region agent log
        if let Ok(s) = tensor_stats("latents_final", &latents) {
            debug_log("A", "pipeline.rs:final", "final latents before VAE", s);
        }
        // #endregion

        let latents_for_vae = latents
            .to_dtype(self.vae_dtype)?
            .to_device(&self.vae_device)?;
        let frames = self.vae.decode(&latents_for_vae)?;
        // #region agent log
        if let Ok(s) = tensor_stats("vae_output", &frames) {
            debug_log("B", "pipeline.rs:vae", "VAE decode output", s);
        }
        // #endregion
        let frames = frames.to_device(&self.device)?;
        Ok(WanPipelineOutput { frames })
    }
}

/// Loaded Wan T2V pipeline (single transformer stage).
pub struct WanPipeline {
    config: WanFullConfig,
    tokenizer: WanTokenizer,
    text_encoder: Umt5TextEncoder,
    stack: WanDenoiseStack,
}

impl WanPipeline {
    pub fn load(
        root: impl AsRef<Path>,
        device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        Self::load_with_devices(root, device, device, device, transformer_dtype, vae_dtype)
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
        )
    }

    /// Load pipeline with per-component devices (e.g. VAE on CPU to save VRAM).
    pub fn load_with_devices(
        root: impl AsRef<Path>,
        device: &Device,
        text_encoder_device: &Device,
        vae_device: &Device,
        transformer_dtype: DType,
        vae_dtype: DType,
    ) -> std::result::Result<Self, crate::models::ltx_video::loader::LoaderError> {
        let paths = WanComponentPaths::from_root(&root);
        paths.validate_exists()?;
        let config = load_wan_config(&root)?;

        let tokenizer = WanTokenizer::from_pretrained(&paths.tokenizer_dir)?;
        // Load denoise stack on GPU first so UMT5 never shares VRAM with transformer.
        let stack =
            WanDenoiseStack::load_with_vae_device(&root, device, vae_device, transformer_dtype, vae_dtype)?;
        let text_encoder = Umt5TextEncoder::load(
            &paths.root.join("text_encoder"),
            text_encoder_device,
            transformer_dtype,
        )?;

        Ok(Self {
            config,
            tokenizer,
            text_encoder,
            stack,
        })
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
        (num_frames - 1) / self.config.temporal_compression() + 1
    }

    pub fn latent_spatial_size(&self, pixel_size: usize) -> usize {
        pixel_size / self.config.spatial_compression()
    }

    pub fn latent_shape(&self, batch: usize, height: usize, width: usize, num_frames: usize) -> Vec<usize> {
        latent_shape_from_config(&self.config, batch, height, width, num_frames)
    }

    pub fn check_inputs(&self, req: &WanGenerateRequest) -> Result<()> {
        if req.height % 16 != 0 || req.width % 16 != 0 {
            candle_core::bail!(
                "height and width must be divisible by 16, got {}x{}",
                req.height,
                req.width
            );
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
            candle_core::bail!("provide either negative_prompt or negative_prompt_embeds, not both");
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

        let prompt = req.prompt.as_ref().expect("prompt checked");
        let cleaned = prompt_clean(prompt);
        let prompts = vec![cleaned];
        let negative = req.negative_prompt.as_ref().map(|n| vec![prompt_clean(n)]);

        let te_device = self.text_encoder.device().clone();
        let infer_device = self.device().clone();
        let (prompt_embeds, negative_embeds) = encode_prompt(
            &self.tokenizer,
            &mut self.text_encoder,
            &prompts,
            negative.as_ref().map(|v| v.as_slice()),
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
        self.check_inputs(&req)?;

        let (height, width) = self.round_dimensions(req.height, req.width);
        let do_cfg = self.do_classifier_free_guidance(req.guidance_scale);

        let (prompt_embeds, negative_embeds) = self.encode_prompts(&req, do_cfg)?;

        self.stack.denoise_and_decode(
            height,
            width,
            req.num_frames,
            req.num_inference_steps,
            req.guidance_scale,
            req.seed,
            req.initial_latents,
            prompt_embeds,
            negative_embeds,
        )
    }
}

fn normalize_num_frames(num_frames: usize, temporal_compression: usize) -> usize {
    let adjusted = if (num_frames.saturating_sub(1)) % temporal_compression != 0 {
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
    }

    #[test]
    fn normalize_num_frames_rounds_to_temporal_step() {
        assert_eq!(normalize_num_frames(82, 4), 81);
        assert_eq!(normalize_num_frames(81, 4), 81);
        assert_eq!(normalize_num_frames(1, 4), 1);
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
