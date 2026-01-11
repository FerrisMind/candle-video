// Rust 2024
// Минимально самодостаточный порт логики из pipeline_wan.py на candle-тензорах.
// Реальные реализации Tokenizer/TextEncoder/Transformer/VAE/Scheduler подключаются снаружи.

use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;
use regex::Regex;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WanError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, WanError>;

#[derive(Clone, Debug)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputType {
    /// Вернуть латенты (аналог output_type == "latent" в python).
    Latent,
    /// Вернуть тензор видео (аналог "np"/"pt" и т.п. — здесь просто Tensor).
    Tensor,
}

#[derive(Debug)]
pub struct WanPipelineOutput {
    pub frames: Tensor,
}

#[derive(Clone, Debug, Default)]
pub struct AttentionKwargs {
    // В оригинале это прокидывается в attention processor; здесь оставлено как расширяемое место.
}

pub trait PipelineCallback {
    /// Вернуть (возможно) заменённые тензоры.
    fn on_step_end(
        &mut self,
        step: usize,
        timestep: f64,
        latents: Tensor,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Option<Tensor>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)>;
}

pub trait WanTokenizer: Send + Sync {
    /// Возвращает (input_ids, attention_mask), оба размером [batch, max_len].
    fn encode(&self, texts: &[String], max_len: usize) -> Result<(Vec<Vec<u32>>, Vec<Vec<u8>>)>;
}

pub trait WanTextEncoder: Send + Sync {
    /// Возвращает last_hidden_state размером [batch, seq, hidden].
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;
    fn dtype(&self) -> DType;
}

pub trait WanTransformer3DModel: Send + Sync {
    fn dtype(&self) -> DType;
    fn in_channels(&self) -> usize;

    /// hidden_states: [b, c, t, h, w]
    /// timestep: [b] или [b, seq_len] при expand_timesteps
    /// encoder_hidden_states: [b, seq, hidden]
    fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        attention_kwargs: Option<&AttentionKwargs>,
    ) -> Result<Tensor>;
}

pub trait WanVae: Send + Sync {
    fn dtype(&self) -> DType;
    fn z_dim(&self) -> usize;
    fn latents_mean(&self) -> &[f32];
    fn latents_std(&self) -> &[f32];
    fn scale_factor_temporal(&self) -> usize;
    fn scale_factor_spatial(&self) -> usize;

    /// latents: [b, z_dim, t, h, w] -> video tensor
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}

pub trait FlowMatchScheduler: Send + Sync {
    fn order(&self) -> usize;
    fn num_train_timesteps(&self) -> usize;

    fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()>;
    fn timesteps(&self) -> &[f64];

    /// Возвращает latents_{t-1}.
    fn step(&self, noise_pred: &Tensor, timestep: f64, latents: &Tensor) -> Result<Tensor>;
}

/// Упрощённый постпроцессор (в оригинале есть разные форматы output_type).
#[derive(Clone, Debug)]
pub struct VideoProcessor {
    #[allow(dead_code)]
    vae_scale_factor_spatial: usize,
}

impl VideoProcessor {
    pub fn new(vae_scale_factor_spatial: usize) -> Self {
        Self {
            vae_scale_factor_spatial,
        }
    }

    pub fn postprocess_video(&self, video: Tensor, _output_type: OutputType) -> Result<Tensor> {
        Ok(video)
    }
}

// ---------- text cleaning (basic_clean/whitespace_clean/prompt_clean) ----------

fn ws_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s+").expect("regex must compile"))
}

fn html_unescape_twice(s: &str) -> String {
    // В python: html.unescape(html.unescape(text))
    let once = html_escape::decode_html_entities(s).to_string();
    html_escape::decode_html_entities(&once).to_string()
}

pub fn basic_clean(text: &str) -> String {
    // В оригинале ftfy опционален; здесь no-op по смыслу.
    html_unescape_twice(text).trim().to_string()
}

pub fn whitespace_clean(text: &str) -> String {
    ws_re().replace_all(text, " ").trim().to_string()
}

pub fn prompt_clean(text: &str) -> String {
    whitespace_clean(&basic_clean(text))
}

// ---------- randn_tensor (аналог randn_tensor из diffusers) ----------

fn randn_tensor_f32(
    shape: &[usize],
    device: &Device,
    seed: Option<u64>,
) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut data = Vec::<f32>::with_capacity(numel);

    match seed {
        Some(seed) => {
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..numel {
                let v: f64 = rng.sample(StandardNormal);
                data.push(v as f32);
            }
        }
        None => {
            let mut rng = rand::thread_rng();
            for _ in 0..numel {
                let v: f64 = rng.sample(StandardNormal);
                data.push(v as f32);
            }
        }
    }

    let t = Tensor::from_vec(data, Shape::from_dims(shape), device)?;
    Ok(t)
}

// ---------- pipeline ----------

pub struct WanPipeline {
    tokenizer: Box<dyn WanTokenizer>,
    text_encoder: Box<dyn WanTextEncoder>,
    vae: Box<dyn WanVae>,
    scheduler: Box<dyn FlowMatchScheduler>,
    transformer: Option<Box<dyn WanTransformer3DModel>>,
    transformer_2: Option<Box<dyn WanTransformer3DModel>>,

    boundary_ratio: Option<f64>,
    expand_timesteps: bool,

    vae_scale_factor_temporal: usize,
    vae_scale_factor_spatial: usize,
    video_processor: VideoProcessor,

    // runtime fields (как в python properties)
    guidance_scale: f64,
    guidance_scale_2: Option<f64>,
    num_timesteps: usize,
    current_timestep: Option<f64>,
    interrupt: bool,
    attention_kwargs: Option<AttentionKwargs>,
}

impl WanPipeline {
    pub fn new(
        tokenizer: Box<dyn WanTokenizer>,
        text_encoder: Box<dyn WanTextEncoder>,
        vae: Box<dyn WanVae>,
        scheduler: Box<dyn FlowMatchScheduler>,
        transformer: Option<Box<dyn WanTransformer3DModel>>,
        transformer_2: Option<Box<dyn WanTransformer3DModel>>,
        boundary_ratio: Option<f64>,
        expand_timesteps: bool,
    ) -> Self {
        let vae_scale_factor_temporal = vae.scale_factor_temporal();
        let vae_scale_factor_spatial = vae.scale_factor_spatial();

        Self {
            tokenizer,
            text_encoder,
            vae,
            scheduler,
            transformer,
            transformer_2,
            boundary_ratio,
            expand_timesteps,
            vae_scale_factor_temporal,
            vae_scale_factor_spatial,
            video_processor: VideoProcessor::new(vae_scale_factor_spatial),
            guidance_scale: 1.0,
            guidance_scale_2: None,
            num_timesteps: 0,
            current_timestep: None,
            interrupt: false,
            attention_kwargs: None,
        }
    }

    fn do_classifier_free_guidance(&self) -> bool {
        self.guidance_scale > 1.0
    }

    fn get_t5_prompt_embeds(
        &self,
        prompt: &[String],
        num_videos_per_prompt: usize,
        max_sequence_length: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let prompt: Vec<String> = prompt.iter().map(|p| prompt_clean(p)).collect();
        let batch_size = prompt.len();

        let (input_ids, attn_mask) = self.tokenizer.encode(&prompt, max_sequence_length)?;
        if input_ids.len() != batch_size || attn_mask.len() != batch_size {
            return Err(WanError::InvalidArgument(
                "tokenizer returned wrong batch size".to_string(),
            ));
        }

        // [b, max_len]
        let mut flat_ids: Vec<u32> = Vec::with_capacity(batch_size * max_sequence_length);
        let mut flat_mask: Vec<u8> = Vec::with_capacity(batch_size * max_sequence_length);
        let mut seq_lens: Vec<usize> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            if input_ids[i].len() != max_sequence_length || attn_mask[i].len() != max_sequence_length {
                return Err(WanError::InvalidArgument(
                    "tokenizer must return max_length-padded sequences".to_string(),
                ));
            }
            flat_ids.extend_from_slice(&input_ids[i]);

            let m = &attn_mask[i];
            flat_mask.extend_from_slice(m);
            let sl = m.iter().filter(|&&v| v > 0).count();
            seq_lens.push(sl);
        }

        let input_ids = Tensor::from_vec(
            flat_ids.into_iter().map(|v| v as i64).collect::<Vec<_>>(),
            (batch_size, max_sequence_length),
            device,
        )?;
        let attention_mask = Tensor::from_vec(
            flat_mask.into_iter().map(|v| v as i64).collect::<Vec<_>>(),
            (batch_size, max_sequence_length),
            device,
        )?;

        // text_encoder(...) -> last_hidden_state: [b, max_len, hidden]
        let prompt_embeds = self.text_encoder.encode(&input_ids, &attention_mask)?;
        let prompt_embeds = prompt_embeds.to_dtype(dtype)?;

        // Обрезаем до seq_lens и дополняем нулями до max_sequence_length (как в python).
        let mut padded: Vec<Tensor> = Vec::with_capacity(batch_size);
        for (i, &sl) in seq_lens.iter().enumerate() {
            let u = prompt_embeds.i((i as i64, .., ..))?; // [max_len, hidden]
            let u = if sl < max_sequence_length {
                let head = u.narrow(0, 0, sl)?; // [sl, hidden]
                let hidden = head.dims()[1];
                let pad = Tensor::zeros((max_sequence_length - sl, hidden), dtype, device)?;
                Tensor::cat(&[&head, &pad], 0)?
            } else {
                u
            };
            padded.push(u);
        }
        let prompt_embeds = Tensor::stack(&padded.iter().collect::<Vec<_>>(), 0)?; // [b, max_len, hidden]

        // Дублируем на num_videos_per_prompt как в torch.repeat/view (простым cat по batch).
        if num_videos_per_prompt == 1 {
            return Ok(prompt_embeds);
        }
        let mut reps: Vec<Tensor> = Vec::with_capacity(batch_size * num_videos_per_prompt);
        for i in 0..batch_size {
            let u = prompt_embeds.i((i as i64, .., ..))?; // [max_len, hidden]
            for _ in 0..num_videos_per_prompt {
                reps.push(u.unsqueeze(0)?); // [1, max_len, hidden]
            }
        }
        let out = Tensor::cat(&reps.iter().collect::<Vec<_>>(), 0)?; // [b*num_videos, max_len, hidden]
        Ok(out)
    }

    pub fn encode_prompt(
        &self,
        prompt: PromptInput,
        negative_prompt: Option<PromptInput>,
        do_classifier_free_guidance: bool,
        num_videos_per_prompt: usize,
        prompt_embeds: Option<Tensor>,
        negative_prompt_embeds: Option<Tensor>,
        max_sequence_length: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let prompt_vec = prompt.into_vec();
        let batch_size = if let Some(ref pe) = prompt_embeds {
            pe.dims()[0]
        } else {
            prompt_vec.len()
        };

        let prompt_embeds = match prompt_embeds {
            Some(pe) => pe,
            None => self.get_t5_prompt_embeds(
                &prompt_vec,
                num_videos_per_prompt,
                max_sequence_length,
                device,
                dtype,
            )?,
        };

        if !do_classifier_free_guidance {
            return Ok((prompt_embeds, None));
        }

        let negative_prompt_embeds = match negative_prompt_embeds {
            Some(ne) => Some(ne),
            None => {
                let neg = negative_prompt.unwrap_or(PromptInput::Single(String::new()));
                let neg_vec = neg.into_vec();

                if neg_vec.len() == 1 && batch_size > 1 {
                    // В python: batch_size * [negative_prompt] если строка.
                    let rep = vec![neg_vec[0].clone(); batch_size];
                    Some(self.get_t5_prompt_embeds(
                        &rep,
                        num_videos_per_prompt,
                        max_sequence_length,
                        device,
                        dtype,
                    )?)
                } else if neg_vec.len() != batch_size {
                    return Err(WanError::InvalidArgument(format!(
                        "`negative_prompt` batch size {} must match prompt batch size {}",
                        neg_vec.len(),
                        batch_size
                    )));
                } else {
                    Some(self.get_t5_prompt_embeds(
                        &neg_vec,
                        num_videos_per_prompt,
                        max_sequence_length,
                        device,
                        dtype,
                    )?)
                }
            }
        };

        Ok((prompt_embeds, negative_prompt_embeds))
    }

    pub fn check_inputs(
        &self,
        prompt_provided: bool,
        negative_prompt_provided: bool,
        height: usize,
        width: usize,
        prompt_embeds_provided: bool,
        negative_prompt_embeds_provided: bool,
        guidance_scale_2: Option<f64>,
    ) -> Result<()> {
        if height % 16 != 0 || width % 16 != 0 {
            return Err(WanError::InvalidArgument(format!(
                "`height` and `width` must be divisible by 16 but are {height} and {width}"
            )));
        }

        if prompt_provided && prompt_embeds_provided {
            return Err(WanError::InvalidArgument(
                "cannot pass both `prompt` and `prompt_embeds`".to_string(),
            ));
        }
        if negative_prompt_provided && negative_prompt_embeds_provided {
            return Err(WanError::InvalidArgument(
                "cannot pass both `negative_prompt` and `negative_prompt_embeds`".to_string(),
            ));
        }
        if !prompt_provided && !prompt_embeds_provided {
            return Err(WanError::InvalidArgument(
                "provide either `prompt` or `prompt_embeds`".to_string(),
            ));
        }

        if self.boundary_ratio.is_none() && guidance_scale_2.is_some() {
            return Err(WanError::InvalidArgument(
                "`guidance_scale_2` is only supported when `boundary_ratio` is set".to_string(),
            ));
        }
        Ok(())
    }

    pub fn prepare_latents(
        &self,
        batch_size: usize,
        num_channels_latents: usize,
        height: usize,
        width: usize,
        num_frames: usize,
        device: &Device,
        latents: Option<Tensor>,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        if let Some(l) = latents {
            return Ok(l);
        }

        let num_latent_frames = (num_frames - 1) / self.vae_scale_factor_temporal + 1;
        let latent_h = height / self.vae_scale_factor_spatial;
        let latent_w = width / self.vae_scale_factor_spatial;

        let shape = [
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_h,
            latent_w,
        ];
        randn_tensor_f32(&shape, device, seed)
    }

    fn pick_model_and_scale(&self, t: f64, boundary_timestep: Option<f64>) -> Result<(&dyn WanTransformer3DModel, f64)> {
        match boundary_timestep {
            None => self.transformer.as_deref().or(self.transformer_2.as_deref())
                .map(|m| (m, self.guidance_scale))
                .ok_or_else(|| WanError::InvalidArgument("no transformer provided".to_string())),
            Some(bt) => {
                if t >= bt {
                    self.transformer
                        .as_deref()
                        .map(|m| (m, self.guidance_scale))
                        .ok_or_else(|| WanError::InvalidArgument("transformer is required for high-noise stage".to_string()))
                } else {
                    let gs2 = self.guidance_scale_2.unwrap_or(self.guidance_scale);
                    self.transformer_2
                        .as_deref()
                        .map(|m| (m, gs2))
                        .ok_or_else(|| WanError::InvalidArgument("transformer_2 is required for low-noise stage".to_string()))
                }
            }
        }
    }

    fn make_timestep_tensor(
        &self,
        t: f64,
        batch: usize,
        num_latent_frames: usize,
        latent_h: usize,
        latent_w: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if !self.expand_timesteps {
            // [batch]
            let data = vec![t as f32; batch];
            return Ok(Tensor::from_vec(data, (batch,), device)?);
        }

        // В python: (mask[0][0][:, ::2, ::2] * t).flatten() => т.к. mask=ones, это просто вектор t.
        // seq_len = num_latent_frames * (latent_h/2) * (latent_w/2)
        let seq_len = num_latent_frames * (latent_h / 2) * (latent_w / 2);
        let row = vec![t as f32; seq_len];
        let mut data = Vec::with_capacity(batch * seq_len);
        for _ in 0..batch {
            data.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(data, (batch, seq_len), device)?)
    }

    pub fn call(
        &mut self,
        prompt: Option<PromptInput>,
        negative_prompt: Option<PromptInput>,
        height: usize,
        width: usize,
        mut num_frames: usize,
        num_inference_steps: usize,
        guidance_scale: f64,
        guidance_scale_2: Option<f64>,
        num_videos_per_prompt: usize,
        device: &Device,
        latents: Option<Tensor>,
        prompt_embeds: Option<Tensor>,
        negative_prompt_embeds: Option<Tensor>,
        output_type: OutputType,
        attention_kwargs: Option<AttentionKwargs>,
        callback: Option<&mut dyn PipelineCallback>,
        max_sequence_length: usize,
        seed: Option<u64>,
    ) -> Result<WanPipelineOutput> {
        self.check_inputs(
            prompt.is_some(),
            negative_prompt.is_some(),
            height,
            width,
            prompt_embeds.is_some(),
            negative_prompt_embeds.is_some(),
            guidance_scale_2,
        )?;

        // num_frames коррекция: (num_frames - 1) % vae_scale_factor_temporal == 0
        if num_frames % self.vae_scale_factor_temporal != 1 {
            num_frames = (num_frames / self.vae_scale_factor_temporal) * self.vae_scale_factor_temporal + 1;
            num_frames = num_frames.max(1);
        }

        self.guidance_scale = guidance_scale;
        self.guidance_scale_2 = if self.boundary_ratio.is_some() && guidance_scale_2.is_none() {
            Some(guidance_scale)
        } else {
            guidance_scale_2
        };
        self.attention_kwargs = attention_kwargs;
        self.current_timestep = None;
        self.interrupt = false;

        // batch size
        let batch_size = if let Some(ref p) = prompt {
            p.clone().into_vec().len()
        } else if let Some(ref pe) = prompt_embeds {
            pe.dims()[0]
        } else {
            return Err(WanError::InvalidArgument("missing prompt and prompt_embeds".to_string()));
        };

        // encode prompt
        let text_dtype = self.text_encoder.dtype();
        let prompt_input = prompt.unwrap_or_else(|| PromptInput::Batch(vec![]));
        let (mut prompt_embeds, mut negative_prompt_embeds) = self.encode_prompt(
            prompt_input,
            negative_prompt,
            self.do_classifier_free_guidance(),
            num_videos_per_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            max_sequence_length,
            device,
            text_dtype,
        )?;

        // transformer dtype selection
        let transformer_dtype = self
            .transformer
            .as_deref()
            .or(self.transformer_2.as_deref())
            .ok_or_else(|| WanError::InvalidArgument("no transformer provided".to_string()))?
            .dtype();

        prompt_embeds = prompt_embeds.to_dtype(transformer_dtype)?;
        if let Some(ref ne) = negative_prompt_embeds {
            negative_prompt_embeds = Some(ne.to_dtype(transformer_dtype)?);
        }

        // timesteps
        self.scheduler.set_timesteps(num_inference_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();
        self.num_timesteps = timesteps.len();

        // latents
        let num_channels_latents = self
            .transformer
            .as_deref()
            .or(self.transformer_2.as_deref())
            .ok_or_else(|| WanError::InvalidArgument("no transformer provided".to_string()))?
            .in_channels();

        let effective_batch = batch_size * num_videos_per_prompt;
        let mut latents = self.prepare_latents(
            effective_batch,
            num_channels_latents,
            height,
            width,
            num_frames,
            device,
            latents,
            seed,
        )?;

        let latent_h = height / self.vae_scale_factor_spatial;
        let latent_w = width / self.vae_scale_factor_spatial;
        let num_latent_frames = (num_frames - 1) / self.vae_scale_factor_temporal + 1;

        // boundary
        let boundary_timestep = self.boundary_ratio.map(|r| r * (self.scheduler.num_train_timesteps() as f64));

        // denoising loop
        for (i, t) in timesteps.iter().copied().enumerate() {
            if self.interrupt {
                continue;
            }
            self.current_timestep = Some(t);

            let (current_model, current_guidance_scale) = self.pick_model_and_scale(t, boundary_timestep)?;

            let latent_model_input = latents.to_dtype(current_model.dtype())?;
            let timestep_t = self.make_timestep_tensor(
                t,
                latent_model_input.dims()[0],
                num_latent_frames,
                latent_h,
                latent_w,
                device,
            )?;

            // cond
            let mut noise_pred = current_model.forward(
                &latent_model_input,
                &timestep_t,
                &prompt_embeds,
                self.attention_kwargs.as_ref(),
            )?;

            // uncond + CFG
            if self.do_classifier_free_guidance() {
                let neg = negative_prompt_embeds
                    .as_ref()
                    .ok_or_else(|| WanError::InvalidArgument("negative_prompt_embeds required for CFG".to_string()))?;
                let noise_uncond = current_model.forward(
                    &latent_model_input,
                    &timestep_t,
                    neg,
                    self.attention_kwargs.as_ref(),
                )?;

                // noise_pred = uncond + scale * (cond - uncond)
                noise_pred = (&noise_uncond + ((&noise_pred - &noise_uncond)? * current_guidance_scale)?)?;
            }

            // scheduler step
            latents = self.scheduler.step(&noise_pred, t, &latents)?;

            // callback
            if let Some(cb) = callback {
                let (l, pe, ne) = cb.on_step_end(
                    i,
                    t,
                    latents,
                    prompt_embeds,
                    negative_prompt_embeds,
                )?;
                latents = l;
                prompt_embeds = pe;
                negative_prompt_embeds = ne;
            }
        }

        self.current_timestep = None;

        // decode / output
        let out = if output_type == OutputType::Latent {
            latents
        } else {
            let latents = latents.to_dtype(self.vae.dtype())?;

            // В python: latents = latents / (1/std) + mean == latents * std + mean
            let z = self.vae.z_dim();
            let mean = Tensor::from_vec(
                self.vae.latents_mean().to_vec(),
                (1, z, 1, 1, 1),
                device,
            )?
            .to_dtype(self.vae.dtype())?;

            let std = Tensor::from_vec(
                self.vae.latents_std().to_vec(),
                (1, z, 1, 1, 1),
                device,
            )?
            .to_dtype(self.vae.dtype())?;

            let latents = ((&latents * &std)? + &mean)?;
            let video = self.vae.decode(&latents)?;
            self.video_processor.postprocess_video(video, output_type)?
        };

        Ok(WanPipelineOutput { frames: out })
    }
}
