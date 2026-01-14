use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::{debug, info};

use crate::interfaces::conditioning::Conditioning;
use crate::interfaces::pipeline::{DiffusionPipeline, PipelineInference, apply_pipeline_io};
use crate::interfaces::video_types::{VideoLatents, VideoLayout};

use super::{
    AutoencoderKLTemporalDecoder, ClipVisionModelWithProjection, EulerDiscreteScheduler, SvdConfig,
    SvdInferenceConfig, UNetSpatioTemporalConditionModel, normalize_for_clip,
};

#[allow(dead_code)]
fn dump_tensor(name: &str, tensor: &Tensor) {
    if std::env::var("DUMP_TENSORS").is_ok() {
        let dir = std::path::Path::new("output/rust_tensors");
        std::fs::create_dir_all(dir).ok();

        if let Ok(t) = tensor.to_dtype(DType::F32)
            && let Ok(flat) = t.flatten_all()
            && let Ok(data) = flat.to_vec1::<f32>()
        {
            let shape: Vec<usize> = t.dims().to_vec();
            let shape_path = dir.join(format!("{}.shape", name));
            let shape_str = shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            std::fs::write(&shape_path, shape_str).ok();

            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{}.bin", name)), bytes).ok();
            debug!("Dumped tensor {} shape={:?}", name, shape);
        }
    }
}

impl DiffusionPipeline for SvdPipeline {}

impl PipelineInference for SvdPipeline {
    fn encode_prompt(
        &mut self,
        _prompt: &str,
        _negative: Option<&str>,
        _device: &Device,
    ) -> Result<Option<Conditioning>> {
        Ok(None)
    }

    fn check_inputs(&self, height: usize, width: usize, num_frames: usize) -> Result<()> {
        if !height.is_multiple_of(8) || !width.is_multiple_of(8) {
            candle_core::bail!(
                "`height` and `width` must be divisible by 8, got {height} and {width}"
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
            let tensor = latents.tensor.to_device(device)?.to_dtype(dtype)?;
            return Ok(VideoLatents { tensor, ..latents });
        }

        let latent_height = height / 8;
        let latent_width = width / 8;
        let tensor = Tensor::randn(
            0f32,
            1f32,
            (
                batch_size * num_frames,
                num_channels_latents,
                latent_height,
                latent_width,
            ),
            device,
        )?
        .to_dtype(dtype)?;

        Ok(VideoLatents {
            tensor,
            layout: VideoLayout::BfCHW,
            batch: batch_size,
            frames: num_frames,
            channels: num_channels_latents,
            height: latent_height,
            width: latent_width,
        })
    }

    fn guidance_scale(&self) -> f64 {
        self.guidance_scale
    }

    fn do_classifier_free_guidance(&self) -> bool {
        self.do_classifier_free_guidance
    }

    fn num_timesteps(&self) -> usize {
        self.num_timesteps
    }
}

pub struct SvdPipeline {
    unet: UNetSpatioTemporalConditionModel,
    vae: AutoencoderKLTemporalDecoder,
    image_encoder: ClipVisionModelWithProjection,
    scheduler: EulerDiscreteScheduler,
    device: Device,
    dtype: DType,
    guidance_scale: f64,
    do_classifier_free_guidance: bool,
    num_timesteps: usize,
}

impl SvdPipeline {
    pub fn new(vb: VarBuilder, config: &SvdConfig, device: Device, dtype: DType) -> Result<Self> {
        let unet = UNetSpatioTemporalConditionModel::new(vb.pp("unet"), &config.unet)?;
        let vae = AutoencoderKLTemporalDecoder::new(vb.pp("vae"), &config.vae)?;
        let image_encoder =
            ClipVisionModelWithProjection::new(vb.pp("image_encoder"), &config.clip)?;
        let scheduler = EulerDiscreteScheduler::new(config.scheduler.clone());

        let mut pipeline = Self {
            unet,
            vae,
            image_encoder,
            scheduler,
            device,
            dtype,
            guidance_scale: 1.0,
            do_classifier_free_guidance: false,
            num_timesteps: 0,
        };

        DiffusionPipeline::register_modules(
            &mut pipeline,
            &["unet", "vae", "image_encoder", "scheduler"],
        );
        apply_pipeline_io(&mut pipeline);

        Ok(pipeline)
    }

    pub fn new_from_parts(
        unet_vb: VarBuilder,
        vae_vb: VarBuilder,
        clip_vb: VarBuilder,
        config: &SvdConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let unet = UNetSpatioTemporalConditionModel::new(unet_vb, &config.unet)?;
        let vae = AutoencoderKLTemporalDecoder::new(vae_vb, &config.vae)?;
        let image_encoder = ClipVisionModelWithProjection::new(clip_vb, &config.clip)?;
        let scheduler = EulerDiscreteScheduler::new(config.scheduler.clone());

        let mut pipeline = Self {
            unet,
            vae,
            image_encoder,
            scheduler,
            device,
            dtype,
            guidance_scale: 1.0,
            do_classifier_free_guidance: false,
            num_timesteps: 0,
        };

        DiffusionPipeline::register_modules(
            &mut pipeline,
            &["unet", "vae", "image_encoder", "scheduler"],
        );
        apply_pipeline_io(&mut pipeline);

        Ok(pipeline)
    }

    pub fn generate(&mut self, image: &Tensor, config: &SvdInferenceConfig) -> Result<Tensor> {
        let batch_size = 1;
        let num_frames = config.num_frames;
        let height = config.height;
        let width = config.width;
        let latent_height = height / 8;
        let latent_width = width / 8;

        PipelineInference::check_inputs(self, height, width, num_frames)?;
        self.guidance_scale = config.max_guidance_scale;
        self.do_classifier_free_guidance = config.max_guidance_scale > 1.0;
        self.num_timesteps = config.num_inference_steps;
        let guidance_scale = PipelineInference::guidance_scale(self);
        let do_cfg = PipelineInference::do_classifier_free_guidance(self);
        let num_timesteps = PipelineInference::num_timesteps(self);
        if guidance_scale < 0.0 {
            candle_core::bail!("guidance_scale must be >= 0");
        }
        if num_timesteps == 0 {
            candle_core::bail!("num_timesteps must be > 0");
        }

        info!(
            num_frames,
            height, width, latent_height, latent_width, "SVD generate start"
        );
        debug!(input_shape = ?image.dims(), dtype = ?image.dtype(), "Input image");

        let clip_image = image.interpolate2d(224, 224)?;

        let clip_image_01 = ((clip_image + 1.0)? / 2.0)?;
        let image_normalized = normalize_for_clip(&clip_image_01, &self.device)?;
        let image_embeddings = self.image_encoder.forward(&image_normalized)?;

        let embed_dim = image_embeddings.dim(1)?;
        let image_embeddings = image_embeddings
            .unsqueeze(1)?
            .repeat((1, num_frames, 1))?
            .reshape((batch_size * num_frames, 1, embed_dim))?;
        dump_tensor("image_embeddings_raw", &image_embeddings);

        debug!(image_embeddings_shape = ?image_embeddings.dims(), "CLIP image embeddings");

        let noise_aug_strength = config.noise_aug_strength;
        let noise = Tensor::randn_like(image, 0.0, 1.0)?;
        let noise = noise.affine(noise_aug_strength, 0.0)?;
        let image_augmented = (image + &noise)?;
        dump_tensor("vae_input", &image_augmented);
        let image_latents = self.vae.encode_to_latent(&image_augmented)?;
        dump_tensor("image_latents_raw", &image_latents);

        let image_cond_latents = image_latents
            .unsqueeze(1)?
            .repeat((1, num_frames, 1, 1, 1))?
            .reshape((batch_size * num_frames, 4, latent_height, latent_width))?;

        let num_channels_latents = image_cond_latents.dim(1)?;
        let latents = PipelineInference::prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.dtype,
            &self.device,
            None,
        )?;
        let latents = latents.tensor;

        let added_time_ids = Tensor::new(
            &[[
                config.fps.saturating_sub(1) as f32,
                config.motion_bucket_id as f32,
                noise_aug_strength as f32,
            ]],
            &self.device,
        )?
        .to_dtype(self.dtype)?
        .repeat((batch_size * num_frames, 1))?;
        dump_tensor("added_time_ids", &added_time_ids);

        self.scheduler.set_timesteps(num_timesteps, &self.device)?;
        let timesteps: Vec<f64> = self.scheduler.timesteps().to_vec();

        debug!(latents_shape = ?latents.dims(), image_cond_latents_shape = ?image_cond_latents.dims(), "Initial tensors");

        let mut latents = latents.affine(self.scheduler.init_noise_sigma(), 0.0)?;

        let guidance_scales: Vec<f32> = (0..num_frames)
            .map(|f| {
                let t = if num_frames > 1 {
                    f as f64 / (num_frames - 1) as f64
                } else {
                    0.0
                };
                (config.min_guidance_scale
                    + (config.max_guidance_scale - config.min_guidance_scale) * t)
                    as f32
            })
            .collect();

        let guidance_scale_vec: Vec<f32> = (0..batch_size)
            .flat_map(|_| guidance_scales.iter().copied())
            .collect();
        let guidance_scale_tensor = Tensor::new(guidance_scale_vec.as_slice(), &self.device)?
            .to_dtype(self.dtype)?
            .reshape((batch_size * num_frames, 1, 1, 1))?;

        let do_classifier_free_guidance = do_cfg;

        let (image_cond_latents_cfg, encoder_states_cfg, added_time_ids_cfg) =
            if do_classifier_free_guidance {
                let zeros_cond = image_cond_latents.zeros_like()?;
                let cond_cfg = Tensor::cat(&[&zeros_cond, &image_cond_latents], 0)?;
                drop(zeros_cond);

                let zeros_emb = image_embeddings.zeros_like()?;
                let emb_cfg = Tensor::cat(&[&zeros_emb, &image_embeddings], 0)?;
                drop(zeros_emb);

                let time_ids_cfg = Tensor::cat(&[&added_time_ids, &added_time_ids], 0)?;

                (cond_cfg, emb_cfg, time_ids_cfg)
            } else {
                (
                    image_cond_latents.clone(),
                    image_embeddings.clone(),
                    added_time_ids.clone(),
                )
            };

        info!(
            num_steps = timesteps.len(),
            cfg = do_classifier_free_guidance,
            "Starting denoising loop"
        );

        let total_steps = timesteps.len();
        for (i, &t) in timesteps.iter().enumerate() {
            println!("  Step {}/{} (t={:.4})", i + 1, total_steps, t);
            debug!(step = i, timestep = t, latents_shape = ?latents.dims(), "Denoising step");

            let base_timestep = Tensor::new(&[t], &self.device)?
                .to_dtype(self.dtype)?
                .repeat(batch_size * num_frames)?;

            let noise_pred = if do_classifier_free_guidance {
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

                let latent_model_input =
                    self.scheduler.scale_model_input(&latent_model_input, i)?;

                let latent_input = Tensor::cat(&[&latent_model_input, &image_cond_latents_cfg], 1)?;

                let timestep_cfg = Tensor::cat(&[&base_timestep, &base_timestep], 0)?;

                let noise_pred = self.unet.forward(
                    &latent_input,
                    &timestep_cfg,
                    &encoder_states_cfg,
                    &added_time_ids_cfg,
                    num_frames,
                    None,
                )?;

                let half_batch = batch_size * num_frames;
                let noise_pred_uncond = noise_pred.narrow(0, 0, half_batch)?;
                let noise_pred_cond = noise_pred.narrow(0, half_batch, half_batch)?;

                let diff = (&noise_pred_cond - &noise_pred_uncond)?;
                (&noise_pred_uncond + diff.broadcast_mul(&guidance_scale_tensor)?)?
            } else {
                let latent_model_input = self.scheduler.scale_model_input(&latents, i)?;
                let latent_input = Tensor::cat(&[&latent_model_input, &image_cond_latents], 1)?;
                self.unet.forward(
                    &latent_input,
                    &base_timestep,
                    &image_embeddings,
                    &added_time_ids,
                    num_frames,
                    None,
                )?
            };

            debug!(noise_pred_shape = ?noise_pred.dims(), "Before scheduler step");

            if let Ok(np_f32) = noise_pred.to_dtype(candle_core::DType::F32)
                && let Ok(flat) = np_f32.flatten_all()
            {
                let min = flat.min(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                let max = flat.max(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                println!("    noise_pred: min={:?}, max={:?}", min, max);
            }

            let output = self.scheduler.step(&noise_pred, i, &latents)?;
            latents = output.prev_sample;

            if let Ok(lat_f32) = latents.to_dtype(candle_core::DType::F32)
                && let Ok(flat) = lat_f32.flatten_all()
            {
                let min = flat.min(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                let max = flat.max(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                println!("    latents: min={:?}, max={:?}", min, max);
            }
        }

        info!("Denoising complete, decoding latents");
        debug!(final_latents_shape = ?latents.dims(), "Latents before VAE decode");

        let video_frames = self
            .vae
            .decode(&latents, num_frames, config.decode_chunk_size)?;
        debug!(video_frames_shape = ?video_frames.dims(), "After VAE decode");

        let video_frames = video_frames.reshape((batch_size, num_frames, 3, height, width))?;

        let video_frames = ((video_frames + 1.0)? / 2.0)?;
        let video_frames = video_frames.clamp(0.0, 1.0)?;

        Ok(video_frames)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

pub fn load_image(
    path: &str,
    height: usize,
    width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open image: {}", e)))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read image: {}", e)))?;

    let img = image::load_from_memory(&buffer)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to decode image: {}", e)))?;

    let img = img.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let img = img.to_rgb8();

    let data: Vec<f32> = img
        .pixels()
        .flat_map(|p| {
            let [r, g, b] = p.0;
            [
                (r as f32 / 255.0) * 2.0 - 1.0,
                (g as f32 / 255.0) * 2.0 - 1.0,
                (b as f32 / 255.0) * 2.0 - 1.0,
            ]
        })
        .collect();

    let tensor = Tensor::from_vec(data, (height, width, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(dtype)?;

    Ok(tensor)
}

pub fn save_video_frames(frames: &Tensor, output_dir: &str) -> Result<()> {
    use std::fs;

    fs::create_dir_all(output_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create output dir: {}", e)))?;

    let (batch, num_frames, _channels, height, width) = frames.dims5()?;

    for b in 0..batch {
        for f in 0..num_frames {
            let frame = frames.i((b, f, .., .., ..))?;
            let frame = (frame * 255.0)?.to_dtype(DType::U8)?;

            let frame_data: Vec<u8> = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1()?;

            let img = image::RgbImage::from_raw(width as u32, height as u32, frame_data)
                .ok_or_else(|| candle_core::Error::Msg("Failed to create image".to_string()))?;

            let path = format!("{}/frame_{:04}.png", output_dir, f);
            img.save(&path)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to save frame: {}", e)))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config() {
        let config = SvdInferenceConfig::default();
        assert_eq!(config.num_frames, 14);
        assert_eq!(config.height, 576);
        assert_eq!(config.width, 1024);
    }
}
