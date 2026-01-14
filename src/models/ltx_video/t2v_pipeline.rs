use candle_core::{D, DType, Device, IndexOp, Result, Tensor};

pub use crate::interfaces::autoencoder::VideoAutoencoder;
pub use crate::interfaces::conditioning::{Conditioning, TextConditioner};

pub use crate::interfaces::flow_match_scheduler::{
    Scheduler, SchedulerConfig, TimestepsSpec, calculate_shift, retrieve_timesteps,
};
use crate::interfaces::pipeline::{
    DiffusionPipeline, PipelineInference, PipelineInput, PipelineOutput, apply_pipeline_io,
};
use crate::interfaces::processor::{VideoInput, VideoOutput, VideoProcessor};
pub use crate::interfaces::scheduler::VideoScheduler;
pub use crate::interfaces::video_types::{VideoLatents, VideoLayout};

pub trait Tokenizer {
    fn encode_batch(&self, prompts: &[String], max_length: usize) -> Result<(Tensor, Tensor)>;

    fn model_max_length(&self) -> usize;
}

pub trait TextEncoder: TextConditioner {
    fn dtype(&self) -> DType;

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor>;
}

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub in_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub num_layers: usize,
}

pub trait VideoTransformer3D {
    fn config(&self) -> &TransformerConfig;

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f32, f32, f32)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor>;

    fn set_skip_block_list(&mut self, list: Vec<usize>);
}

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub scaling_factor: f32,
    pub timestep_conditioning: bool,
}

pub trait VaeLtxVideo: VideoAutoencoder {
    fn dtype(&self) -> DType;
    fn spatial_compression_ratio(&self) -> usize;
    fn temporal_compression_ratio(&self) -> usize;
    fn config(&self) -> &VaeConfig;

    fn latents_mean(&self) -> &Tensor;
    fn latents_std(&self) -> &Tensor;

    fn decode(&self, latents: &Tensor, timestep: Option<&Tensor>) -> Result<Tensor>;
}

#[derive(Debug, Clone)]
pub enum PromptInput {
    Single(String),
    Batch(Vec<String>),
}

impl PromptInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            PromptInput::Single(s) => vec![s],
            PromptInput::Batch(v) => v,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    Latent,
    Tensor,
}

#[derive(Debug, Clone)]
pub struct LtxPipelineOutput {
    pub frames: Tensor,
}

pub struct LtxVideoProcessor {
    pub config: VaeConfig,
}

impl LtxVideoProcessor {
    pub fn new(config: VaeConfig) -> Self {
        Self { config }
    }
}

impl VideoProcessor for LtxVideoProcessor {
    fn preprocess_video(&self, input: &VideoInput) -> Result<Tensor> {
        let input: &PipelineInput = input;
        match input {
            VideoInput::Latents(latents) => Ok(latents.tensor.clone()),
            VideoInput::Image(_) | VideoInput::Video(_) => {
                candle_core::bail!("LTX video processor does not support image/video inputs")
            }
        }
    }

    fn postprocess_video(&self, video: Tensor) -> Result<VideoOutput> {
        let video = video.affine(0.5, 0.5)?;
        let video = video.clamp(0.0f32, 1.0f32)?;

        let video = video.affine(255.0, 0.0)?;
        Ok(VideoOutput::Tensor(video))
    }
}

fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 0 {
        return vec![];
    }
    if steps == 1 {
        return vec![start];
    }
    let denom = (steps - 1) as f32;
    (0..steps)
        .map(|i| start + (end - start) * (i as f32) / denom)
        .collect()
}

fn std_over_dims_except0_keepdim(x: &Tensor) -> Result<Tensor> {
    let rank = x.rank();
    if rank < 2 {
        candle_core::bail!("std_over_dims_except0_keepdim expects rank >= 2, got {rank}");
    }
    let b = x.dim(0)?;
    let flat = x.flatten_from(1)?;
    let var = flat.var_keepdim(1)?;
    let std = var.sqrt()?;
    let mut shape = Vec::with_capacity(rank);
    shape.push(b);
    shape.extend(std::iter::repeat_n(1usize, rank - 1));
    std.reshape(shape)
}

pub fn rescale_noise_cfg(
    noise_cfg: &Tensor,
    noise_pred_text: &Tensor,
    guidance_rescale: f32,
) -> Result<Tensor> {
    let std_text = std_over_dims_except0_keepdim(noise_pred_text)?;
    let std_cfg = std_over_dims_except0_keepdim(noise_cfg)?;

    let ratio = std_text.broadcast_div(&std_cfg)?;
    let noise_pred_rescaled = noise_cfg.broadcast_mul(&ratio)?;

    let a = noise_pred_rescaled.affine(guidance_rescale as f64, 0.0)?;
    let b = noise_cfg.affine((1.0 - guidance_rescale) as f64, 0.0)?;
    a.broadcast_add(&b)
}

pub struct LtxPipeline<'a> {
    pub scheduler: Box<dyn Scheduler + 'a>,
    pub vae: Box<dyn VaeLtxVideo + 'a>,
    pub text_encoder: Box<dyn TextEncoder + 'a>,
    pub tokenizer: Box<dyn Tokenizer + 'a>,
    pub transformer: Box<dyn VideoTransformer3D + 'a>,
    pub video_processor: Box<dyn VideoProcessor + 'a>,

    pub tokenizer_max_length: usize,

    pub vae_spatial_compression_ratio: usize,
    pub vae_temporal_compression_ratio: usize,
    pub transformer_spatial_patch_size: usize,
    pub transformer_temporal_patch_size: usize,

    pub guidance_scale: f32,
    pub guidance_rescale: f32,
    pub stg_scale: f32,
    pub num_timesteps: usize,
    pub current_timestep: Option<i64>,
    pub interrupt: bool,
}

impl<'a> LtxPipeline<'a> {
    pub fn new(
        scheduler: Box<dyn Scheduler + 'a>,
        vae: Box<dyn VaeLtxVideo + 'a>,
        text_encoder: Box<dyn TextEncoder + 'a>,
        tokenizer: Box<dyn Tokenizer + 'a>,
        transformer: Box<dyn VideoTransformer3D + 'a>,
        video_processor: Box<dyn VideoProcessor + 'a>,
    ) -> Self {
        let vae_spatial = vae.spatial_compression_ratio();
        let vae_temporal = vae.temporal_compression_ratio();
        let tcfg = transformer.config().clone();
        let max_len = tokenizer.model_max_length();

        let mut pipeline = Self {
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            transformer,
            video_processor,
            tokenizer_max_length: max_len,
            vae_spatial_compression_ratio: vae_spatial,
            vae_temporal_compression_ratio: vae_temporal,
            transformer_spatial_patch_size: tcfg.patch_size,
            transformer_temporal_patch_size: tcfg.patch_size_t,
            guidance_scale: 1.0,
            guidance_rescale: 0.0,
            stg_scale: 1.0,
            num_timesteps: 0,
            current_timestep: None,
            interrupt: false,
        };

        DiffusionPipeline::register_modules(
            &mut pipeline,
            &[
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "transformer",
                "video_processor",
            ],
        );
        apply_pipeline_io(&mut pipeline);

        pipeline
    }

    pub fn do_spatio_temporal_guidance(&self) -> bool {
        self.stg_scale > 0.0
    }

    pub fn do_classifier_free_guidance(&self) -> bool {
        self.guidance_scale > 1.0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check_inputs(
        &self,
        prompt: Option<&PromptInput>,
        height: usize,
        width: usize,
        prompt_embeds: Option<&Tensor>,
        negative_prompt_embeds: Option<&Tensor>,
        prompt_attention_mask: Option<&Tensor>,
        negative_prompt_attention_mask: Option<&Tensor>,
    ) -> Result<()> {
        if !height.is_multiple_of(32) || !width.is_multiple_of(32) {
            candle_core::bail!(
                "`height` and `width` must be divisible by 32, got {height} and {width}"
            );
        }

        if prompt.is_some() && prompt_embeds.is_some() {
            candle_core::bail!("Cannot forward both `prompt` and `prompt_embeds`.");
        }
        if prompt.is_none() && prompt_embeds.is_none() {
            candle_core::bail!("Provide either `prompt` or `prompt_embeds`.");
        }

        if prompt_embeds.is_some() && prompt_attention_mask.is_none() {
            candle_core::bail!(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds`."
            );
        }
        if negative_prompt_embeds.is_some() && negative_prompt_attention_mask.is_none() {
            candle_core::bail!(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`."
            );
        }

        if prompt_embeds
            .zip(negative_prompt_embeds)
            .is_some_and(|(p, n)| p.dims() != n.dims())
        {
            candle_core::bail!(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape."
            );
        }
        if prompt_attention_mask
            .zip(negative_prompt_attention_mask)
            .is_some_and(|(p, n)| p.dims() != n.dims())
        {
            candle_core::bail!(
                "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape."
            );
        }

        Ok(())
    }

    fn get_t5_prompt_embeds(
        &mut self,
        prompt: &[String],
        num_videos_per_prompt: usize,
        max_sequence_length: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = prompt.len();
        let (input_ids, attention_mask) =
            self.tokenizer.encode_batch(prompt, max_sequence_length)?;
        let input_ids = input_ids.to_device(device)?;
        let attention_mask = attention_mask.to_device(device)?;

        let prompt_embeds = self.text_encoder.forward(&input_ids)?;
        let prompt_embeds = prompt_embeds.to_device(device)?.to_dtype(dtype)?;

        let dims = prompt_embeds.dims();
        if dims.len() != 3 {
            candle_core::bail!("text_encoder output must be rank-3 [B,L,D], got {:?}", dims);
        }
        let seq_len = dims[1];
        let hidden = dims[2];

        let pe = prompt_embeds.repeat((1usize, num_videos_per_prompt, 1usize))?;
        let pe = pe.reshape((batch_size * num_videos_per_prompt, seq_len, hidden))?;

        let am = attention_mask.to_dtype(dtype)?.to_device(device)?;
        Ok((pe, am))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_prompt(
        &mut self,
        prompt: PromptInput,
        negative_prompt: Option<PromptInput>,
        do_classifier_free_guidance: bool,
        num_videos_per_prompt: usize,
        prompt_embeds: Option<Tensor>,
        negative_prompt_embeds: Option<Tensor>,
        prompt_attention_mask: Option<Tensor>,
        negative_prompt_attention_mask: Option<Tensor>,
        max_sequence_length: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let prompt_vec = prompt.clone().into_vec();
        let batch_size = if let Some(ref pe) = prompt_embeds {
            pe.dim(0)?
        } else {
            prompt_vec.len()
        };

        let (prompt_embeds, prompt_attention_mask) =
            if let (Some(pe), Some(pm)) = (prompt_embeds, prompt_attention_mask) {
                (pe, pm)
            } else {
                self.get_t5_prompt_embeds(
                    &prompt_vec,
                    num_videos_per_prompt,
                    max_sequence_length,
                    device,
                    dtype,
                )?
            };

        let (negative_prompt_embeds, negative_prompt_attention_mask) =
            if do_classifier_free_guidance && negative_prompt_embeds.is_none() {
                let neg = match negative_prompt {
                    Some(p) => p,
                    None => PromptInput::Single(String::new()),
                };
                let mut neg_vec = neg.into_vec();
                if neg_vec.len() == 1 && batch_size > 1 {
                    neg_vec = vec![neg_vec[0].clone(); batch_size];
                }
                if neg_vec.len() != batch_size {
                    candle_core::bail!(
                        "negative_prompt batch mismatch: expected {batch_size}, got {}",
                        neg_vec.len()
                    );
                }
                self.get_t5_prompt_embeds(
                    &neg_vec,
                    num_videos_per_prompt,
                    max_sequence_length,
                    device,
                    dtype,
                )?
            } else {
                let ne =
                    negative_prompt_embeds.unwrap_or_else(|| prompt_embeds.zeros_like().unwrap());
                let nm = negative_prompt_attention_mask
                    .unwrap_or_else(|| prompt_attention_mask.zeros_like().unwrap());
                (ne, nm)
            };

        Ok((
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ))
    }

    pub fn pack_latents(
        latents: &Tensor,
        patch_size: usize,
        patch_size_t: usize,
    ) -> Result<Tensor> {
        let dims = latents.dims();
        if dims.len() != 5 {
            candle_core::bail!("pack_latents expects [B,C,F,H,W], got {:?}", dims);
        }
        let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);

        if f % patch_size_t != 0 || h % patch_size != 0 || w % patch_size != 0 {
            candle_core::bail!("latents shape not divisible by patch sizes");
        }

        let f2 = f / patch_size_t;
        let h2 = h / patch_size;
        let w2 = w / patch_size;

        let x = latents.reshape(vec![b, c, f2, patch_size_t, h2, patch_size, w2, patch_size])?;

        let x = x.permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?;

        let x = x.flatten_from(4)?;
        let d = x.dim(4)?;

        let s = f2 * h2 * w2;
        x.reshape((b, s, d))
    }

    pub fn unpack_latents(
        latents: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_size_t: usize,
    ) -> Result<Tensor> {
        let dims = latents.dims();
        if dims.len() != 3 {
            candle_core::bail!("unpack_latents expects [B,S,D], got {:?}", dims);
        }
        let b = dims[0];
        let d = dims[2];

        let denom = patch_size_t * patch_size * patch_size;
        if !d.is_multiple_of(denom) {
            candle_core::bail!("D is not divisible by (pt*p*p)");
        }
        let c = d / denom;

        let x = latents.reshape(vec![
            b,
            num_frames,
            height,
            width,
            c,
            patch_size_t,
            patch_size,
            patch_size,
        ])?;

        let x = x.permute(vec![0, 4, 1, 5, 2, 6, 3, 7])?.contiguous()?;

        let x = x.reshape((
            b,
            c,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        ))?;
        Ok(x)
    }

    pub fn normalize_latents(
        latents: &Tensor,
        mean: &Tensor,
        std: &Tensor,
        scaling_factor: f32,
    ) -> Result<Tensor> {
        let c = latents.dim(1)?;
        let mean = mean
            .reshape((1usize, c, 1usize, 1usize, 1usize))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        let std = std
            .reshape((1usize, c, 1usize, 1usize, 1usize))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;

        let x = latents.broadcast_sub(&mean)?;
        let x = x.affine(scaling_factor as f64, 0.0)?.broadcast_div(&std)?;
        Ok(x)
    }

    pub fn denormalize_latents(
        latents: &Tensor,
        mean: &Tensor,
        std: &Tensor,
        scaling_factor: f32,
    ) -> Result<Tensor> {
        let c = latents.dim(1)?;
        let mean = mean
            .reshape((1usize, c, 1usize, 1usize, 1usize))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        let std = std
            .reshape((1usize, c, 1usize, 1usize, 1usize))?
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;

        let x = latents.broadcast_mul(&std)?;
        let x = x
            .affine((1.0 / scaling_factor) as f64, 0.0)?
            .broadcast_add(&mean)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_latents(
        &self,
        batch_size: usize,
        num_channels_latents: usize,
        height: usize,
        width: usize,
        num_frames: usize,
        dtype: DType,
        device: &Device,
        latents: Option<Tensor>,
    ) -> Result<Tensor> {
        if let Some(l) = latents {
            return l.to_device(device)?.to_dtype(dtype);
        }

        let h = height / self.vae_spatial_compression_ratio;
        let w = width / self.vae_spatial_compression_ratio;
        let f = (num_frames - 1) / self.vae_temporal_compression_ratio + 1;

        let shape = (batch_size, num_channels_latents, f, h, w);
        let latents = Tensor::randn(0f32, 1f32, shape, device)?.to_dtype(dtype)?;
        let latents = Self::pack_latents(
            &latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )?;
        Ok(latents)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &mut self,
        prompt: Option<PromptInput>,
        negative_prompt: Option<PromptInput>,
        height: usize,
        width: usize,
        num_frames: usize,
        frame_rate: usize,
        num_inference_steps: usize,
        timesteps: Option<Vec<i64>>,
        sigmas_provided: Option<Vec<f32>>,
        guidance_scale: f32,
        guidance_rescale: f32,
        stg_scale: f32,
        num_videos_per_prompt: usize,
        latents: Option<Tensor>,
        prompt_embeds: Option<Tensor>,
        prompt_attention_mask: Option<Tensor>,
        negative_prompt_embeds: Option<Tensor>,
        negative_prompt_attention_mask: Option<Tensor>,
        decode_timestep: Vec<f32>,
        decode_noise_scale: Option<Vec<f32>>,
        output_type: OutputType,
        max_sequence_length: usize,
        skip_block_list: Option<Vec<usize>>,
        device: &Device,
    ) -> Result<LtxPipelineOutput> {
        PipelineInference::check_inputs(self, height, width, num_frames)?;
        self.check_inputs(
            prompt.as_ref(),
            height,
            width,
            prompt_embeds.as_ref(),
            negative_prompt_embeds.as_ref(),
            prompt_attention_mask.as_ref(),
            negative_prompt_attention_mask.as_ref(),
        )?;

        self.guidance_scale = guidance_scale;
        self.guidance_rescale = guidance_rescale;
        self.stg_scale = stg_scale;
        self.interrupt = false;
        self.current_timestep = None;
        let guidance_scale = PipelineInference::guidance_scale(self);
        let do_cfg = PipelineInference::do_classifier_free_guidance(self);
        if guidance_scale < 0.0 {
            candle_core::bail!("guidance_scale must be >= 0");
        }

        let batch_size = match (&prompt, &prompt_embeds) {
            (Some(PromptInput::Single(_)), _) => 1usize,
            (Some(PromptInput::Batch(v)), _) => v.len(),
            (None, Some(pe)) => pe.dim(0)?,
            _ => candle_core::bail!("Invalid prompt/prompt_embeds combination"),
        };
        let effective_batch = batch_size * num_videos_per_prompt;

        if let Some(ref list) = skip_block_list {
            if !self.do_spatio_temporal_guidance() {
                self.transformer.set_skip_block_list(list.clone());
            } else {
                self.transformer.set_skip_block_list(vec![]);
            }
        }

        let dtype = self.text_encoder.dtype();
        let prompt_in = prompt
            .clone()
            .unwrap_or_else(|| PromptInput::Single(String::new()));
        let conditioning = if num_videos_per_prompt == 1
            && prompt_embeds.is_none()
            && negative_prompt_embeds.is_none()
            && prompt_attention_mask.is_none()
            && negative_prompt_attention_mask.is_none()
        {
            match (&prompt_in, negative_prompt.as_ref()) {
                (PromptInput::Single(p), None) => {
                    PipelineInference::encode_prompt(self, p, None, device)?
                }
                (PromptInput::Single(p), Some(PromptInput::Single(n))) => {
                    PipelineInference::encode_prompt(self, p, Some(n.as_str()), device)?
                }
                _ => None,
            }
        } else {
            None
        };

        let (mut p_emb, mut p_mask, n_emb, n_mask) = if let Some(conditioning) = conditioning {
            let prompt_embeds = conditioning.prompt_embeds;
            let prompt_attention_mask = conditioning.prompt_attention_mask;
            let n_emb = conditioning
                .negative_prompt_embeds
                .unwrap_or(prompt_embeds.zeros_like()?);
            let n_mask = conditioning
                .negative_prompt_attention_mask
                .unwrap_or(prompt_attention_mask.zeros_like()?);
            (prompt_embeds, prompt_attention_mask, n_emb, n_mask)
        } else {
            self.encode_prompt(
                prompt_in,
                negative_prompt,
                do_cfg,
                num_videos_per_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                prompt_attention_mask,
                negative_prompt_attention_mask,
                max_sequence_length,
                device,
                dtype,
            )?
        };

        let prompt_embeds_cond = p_emb.clone();
        let prompt_mask_cond = p_mask.clone();
        let prompt_embeds_uncond = n_emb.clone();
        let prompt_mask_uncond = n_mask.clone();

        if do_cfg {
            p_emb = Tensor::cat(&[n_emb, p_emb], 0)?;
            p_mask = Tensor::cat(&[n_mask, p_mask], 0)?;
        }

        let num_channels_latents = self.transformer.config().in_channels;
        let mut latents = if let Some(latents) = latents {
            self.prepare_latents(
                effective_batch,
                num_channels_latents,
                height,
                width,
                num_frames,
                DType::F32,
                device,
                Some(latents),
            )?
        } else {
            let prepared = PipelineInference::prepare_latents(
                self,
                effective_batch,
                num_channels_latents,
                height,
                width,
                num_frames,
                DType::F32,
                device,
                None,
            )?;
            let prepared = prepared.to_canonical()?;
            Self::pack_latents(
                &prepared.tensor.permute((0, 2, 1, 3, 4))?,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )?
        };

        let latent_num_frames = (num_frames - 1) / self.vae_temporal_compression_ratio + 1;
        let latent_height = height / self.vae_spatial_compression_ratio;
        let latent_width = width / self.vae_spatial_compression_ratio;

        let video_sequence_length = latent_num_frames * latent_height * latent_width;

        let has_custom_sigmas = sigmas_provided.is_some();

        let sigmas = if sigmas_provided.is_none() && timesteps.is_none() {
            Some(linspace(
                1.0,
                1.0 / (num_inference_steps as f32),
                num_inference_steps,
            ))
        } else {
            sigmas_provided
        };

        let scfg = self.scheduler.config().clone();
        let mu = if has_custom_sigmas {
            0.0
        } else {
            calculate_shift(
                video_sequence_length,
                scfg.base_image_seq_len,
                scfg.max_image_seq_len,
                scfg.base_shift,
                scfg.max_shift,
            )
        };

        if !has_custom_sigmas {
            println!(
                "  Calculated SD3 shift (mu): {:.4} for {} tokens",
                mu, video_sequence_length
            );
        } else {
            println!("  Using custom distilled sigmas (mu=0.0)");
        }

        let (ts, _nsteps_effective) = retrieve_timesteps(
            self.scheduler.as_mut(),
            Some(num_inference_steps),
            device,
            timesteps,
            sigmas,
            mu,
        )?;
        self.num_timesteps = ts.len();
        if PipelineInference::num_timesteps(self) == 0 {
            candle_core::bail!("num_timesteps must be > 0");
        }

        let num_warmup_steps = ts
            .len()
            .saturating_sub(num_inference_steps * self.scheduler.order());

        let ts_ratio = self.vae_temporal_compression_ratio as f32;
        let sp_ratio = self.vae_spatial_compression_ratio as f32;

        let grid_f =
            Tensor::arange(0u32, latent_num_frames as u32, device)?.to_dtype(DType::F32)?;
        let grid_h = Tensor::arange(0u32, latent_height as u32, device)?.to_dtype(DType::F32)?;
        let grid_w = Tensor::arange(0u32, latent_width as u32, device)?.to_dtype(DType::F32)?;

        let f = grid_f.reshape((latent_num_frames, 1, 1))?.broadcast_as((
            latent_num_frames,
            latent_height,
            latent_width,
        ))?;
        let h = grid_h.reshape((1, latent_height, 1))?.broadcast_as((
            latent_num_frames,
            latent_height,
            latent_width,
        ))?;
        let w = grid_w.reshape((1, 1, latent_width))?.broadcast_as((
            latent_num_frames,
            latent_height,
            latent_width,
        ))?;

        let video_coords = Tensor::stack(&[f, h, w], 0)?
            .flatten_from(1)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let vf = video_coords.i((.., .., 0))?;
        let vh = video_coords.i((.., .., 1))?;
        let vw = video_coords.i((.., .., 2))?;

        let vf = vf
            .affine(ts_ratio as f64, (1.0 - ts_ratio) as f64)?
            .clamp(0.0f32, 1000.0f32)?
            .affine(1.0 / (frame_rate as f64), 0.0)?;

        let vh = vh.affine(sp_ratio as f64, 0.0)?;
        let vw = vw.affine(sp_ratio as f64, 0.0)?;

        let video_coords = Tensor::stack(&[vf, vh, vw], D::Minus1)?.broadcast_as((
            effective_batch,
            video_sequence_length,
            3,
        ))?;

        let num_conds = if self.do_classifier_free_guidance() && self.do_spatio_temporal_guidance()
        {
            3
        } else if self.do_classifier_free_guidance() || self.do_spatio_temporal_guidance() {
            2
        } else {
            1
        };
        let _video_coords_batch = Tensor::cat(&vec![video_coords.clone(); num_conds], 0)?;

        for (i, &t) in ts.iter().enumerate() {
            if self.interrupt {
                continue;
            }

            self.current_timestep = Some(t);

            println!("Step {}/{}: t={}", i + 1, ts.len(), t);

            let noise_pred =
                if self.do_classifier_free_guidance() || self.do_spatio_temporal_guidance() {
                    let b = latents.dim(0)?;
                    let timestep_t = Tensor::full(t as f32, (b,), device)?;
                    let latents_input = latents.to_dtype(dtype)?;

                    let noise_uncond = if self.do_classifier_free_guidance() {
                        Some(self.transformer.forward(
                            &latents_input,
                            &prompt_embeds_uncond,
                            &timestep_t,
                            &prompt_mask_uncond,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            None,
                            Some(&video_coords),
                            None,
                        )?)
                    } else {
                        None
                    };

                    let noise_text = self.transformer.forward(
                        &latents_input,
                        &prompt_embeds_cond,
                        &timestep_t,
                        &prompt_mask_cond,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        None,
                        Some(&video_coords),
                        None,
                    )?;

                    let noise_perturbed = if self.do_spatio_temporal_guidance() {
                        let num_layers = self.transformer.config().num_layers;

                        let mut mask_data = vec![0.0f32; num_layers * b];
                        if let Some(ref layers_to_skip) = skip_block_list {
                            for &layer_idx in layers_to_skip {
                                if layer_idx < num_layers {
                                    for batch_idx in 0..b {
                                        mask_data[layer_idx * b + batch_idx] = 1.0;
                                    }
                                }
                            }
                        }
                        let stg_mask = Tensor::from_vec(mask_data, (num_layers, b), device)?;

                        Some(self.transformer.forward(
                            &latents_input,
                            &prompt_embeds_cond,
                            &timestep_t,
                            &prompt_mask_cond,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            None,
                            Some(&video_coords),
                            Some(&stg_mask),
                        )?)
                    } else {
                        None
                    };

                    let noise_text = noise_text.to_dtype(DType::F32)?;
                    let mut combined = noise_text.clone();

                    if let Some(uncond) = noise_uncond {
                        let uncond = uncond.to_dtype(DType::F32)?;
                        let diff_cfg = noise_text.broadcast_sub(&uncond)?;
                        combined = uncond
                            .broadcast_add(&diff_cfg.affine(self.guidance_scale as f64, 0.0)?)?;

                        if self.guidance_rescale > 0.0 {
                            combined =
                                rescale_noise_cfg(&combined, &noise_text, self.guidance_rescale)?;
                        }
                    }

                    if let Some(perturbed) = noise_perturbed {
                        let perturbed = perturbed.to_dtype(DType::F32)?;
                        let diff_stg = noise_text.broadcast_sub(&perturbed)?;
                        combined = combined
                            .broadcast_add(&diff_stg.affine(self.stg_scale as f64, 0.0)?)?;
                    }

                    combined
                } else {
                    let b = latents.dim(0)?;
                    let timestep_t = Tensor::full(t as f32, (b,), device)?;
                    let latents_input = latents.to_dtype(p_emb.dtype())?;

                    self.transformer
                        .forward(
                            &latents_input,
                            &p_emb,
                            &timestep_t,
                            &p_mask,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            None,
                            Some(&video_coords),
                            None,
                        )?
                        .to_dtype(DType::F32)?
                };

            latents = Scheduler::step(&mut *self.scheduler, &noise_pred, t, &latents)?;

            if i == ts.len() - 1
                || ((i + 1) > num_warmup_steps && (i + 1) % self.scheduler.order() == 0)
            {
            }
        }

        if output_type == OutputType::Latent {
            return Ok(LtxPipelineOutput { frames: latents });
        }

        println!("  Decoding latents with VAE...");
        let mut latents = Self::unpack_latents(
            &latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )?;

        latents = Self::denormalize_latents(
            &latents,
            self.vae.latents_mean(),
            self.vae.latents_std(),
            self.vae.config().scaling_factor,
        )?;

        latents = latents.to_dtype(p_emb.dtype())?;

        let timestep_opt: Option<Tensor>;
        if !self.vae.config().timestep_conditioning {
            timestep_opt = None;
        } else {
            let dt = if decode_timestep.len() == 1 {
                vec![decode_timestep[0]; effective_batch]
            } else {
                decode_timestep
            };
            if dt.len() != effective_batch {
                candle_core::bail!(
                    "decode_timestep must have len 1 or effective_batch={effective_batch}"
                );
            }

            let dns = match decode_noise_scale {
                None => dt.clone(),
                Some(v) if v.len() == 1 => vec![v[0]; effective_batch],
                Some(v) => v,
            };
            if dns.len() != effective_batch {
                candle_core::bail!(
                    "decode_noise_scale must have len 1 or effective_batch={effective_batch}"
                );
            }

            let timestep =
                Tensor::from_vec(dt, (effective_batch,), device)?.to_dtype(latents.dtype())?;
            let scale = Tensor::from_vec(dns, (effective_batch,), device)?
                .to_dtype(latents.dtype())?
                .reshape((effective_batch, 1usize, 1usize, 1usize, 1usize))?;

            let noise =
                Tensor::randn(0f32, 1f32, latents.dims(), device)?.to_dtype(latents.dtype())?;

            let one_minus = scale.affine(-1.0, 1.0)?;
            let a = latents.broadcast_mul(&one_minus)?;
            let b = noise.broadcast_mul(&scale)?;
            latents = a.broadcast_add(&b)?;

            timestep_opt = Some(timestep);
        }

        latents = latents.to_dtype(self.vae.dtype())?;

        let video = VaeLtxVideo::decode(&*self.vae, &latents, timestep_opt.as_ref())?;
        let video: PipelineOutput = self.video_processor.postprocess_video(video)?;
        let video = match video {
            VideoOutput::Tensor(tensor) => tensor,
            VideoOutput::Frames(_) => {
                candle_core::bail!("LTX pipeline expects tensor output from video processor")
            }
        };

        Ok(LtxPipelineOutput { frames: video })
    }
}

impl<'a> DiffusionPipeline for LtxPipeline<'a> {}

impl<'a> PipelineInference for LtxPipeline<'a> {
    fn encode_prompt(
        &mut self,
        prompt: &str,
        negative: Option<&str>,
        device: &Device,
    ) -> Result<Option<Conditioning>> {
        let prompt_input = PromptInput::Single(prompt.to_string());
        let negative_input = negative.map(|v| PromptInput::Single(v.to_string()));
        let do_cfg = negative.is_some();
        let dtype = self.text_encoder.dtype();
        let max_sequence_length = self.tokenizer_max_length;

        let (prompt_embeds, prompt_attention_mask, neg_embeds, neg_attention_mask) =
            LtxPipeline::encode_prompt(
                self,
                prompt_input,
                negative_input,
                do_cfg,
                1,
                None,
                None,
                None,
                None,
                max_sequence_length,
                device,
                dtype,
            )?;

        Ok(Some(Conditioning {
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds: do_cfg.then_some(neg_embeds),
            negative_prompt_attention_mask: do_cfg.then_some(neg_attention_mask),
        }))
    }

    fn check_inputs(&self, height: usize, width: usize, num_frames: usize) -> Result<()> {
        if !height.is_multiple_of(32) || !width.is_multiple_of(32) {
            candle_core::bail!(
                "`height` and `width` must be divisible by 32, got {height} and {width}"
            );
        }
        if num_frames == 0 {
            candle_core::bail!("`num_frames` must be > 0");
        }
        Ok(())
    }

    fn prepare_latents(
        &self,
        batch_size: usize,
        num_channels_latents: usize,
        height: usize,
        width: usize,
        num_frames: usize,
        dtype: DType,
        device: &Device,
        latents: Option<VideoLatents>,
    ) -> Result<VideoLatents> {
        if let Some(latents) = latents {
            let canonical = latents.to_canonical()?;
            let tensor = canonical.tensor.to_device(device)?.to_dtype(dtype)?;
            return Ok(VideoLatents {
                tensor,
                ..canonical
            });
        }

        let latent_height = height / self.vae_spatial_compression_ratio;
        let latent_width = width / self.vae_spatial_compression_ratio;
        let latent_frames = (num_frames - 1) / self.vae_temporal_compression_ratio + 1;

        let tensor = Tensor::randn(
            0f32,
            1f32,
            (
                batch_size,
                latent_frames,
                num_channels_latents,
                latent_height,
                latent_width,
            ),
            device,
        )?
        .to_dtype(dtype)?;

        Ok(VideoLatents {
            tensor,
            layout: VideoLayout::BFCHW,
            batch: batch_size,
            frames: latent_frames,
            channels: num_channels_latents,
            height: latent_height,
            width: latent_width,
        })
    }

    fn guidance_scale(&self) -> f64 {
        self.guidance_scale as f64
    }

    fn do_classifier_free_guidance(&self) -> bool {
        LtxPipeline::do_classifier_free_guidance(self)
    }

    fn num_timesteps(&self) -> usize {
        self.num_timesteps
    }
}
