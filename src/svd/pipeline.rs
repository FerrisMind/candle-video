//! SVD Pipeline - Image-to-Video Generation
//!
//! Main pipeline for Stable Video Diffusion inference.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::svd::{
    AutoencoderKLTemporalDecoder, ClipVisionModelWithProjection, EulerDiscreteScheduler, SvdConfig,
    SvdInferenceConfig, UNetSpatioTemporalConditionModel, normalize_for_clip,
};

/// SVD Pipeline for image-to-video generation
pub struct SvdPipeline {
    unet: UNetSpatioTemporalConditionModel,
    vae: AutoencoderKLTemporalDecoder,
    image_encoder: ClipVisionModelWithProjection,
    scheduler: EulerDiscreteScheduler,
    device: Device,
    dtype: DType,
}

impl SvdPipeline {
    /// Create a new SVD pipeline from a VarBuilder (expects prefixed keys like unet.*, vae.*)
    pub fn new(vb: VarBuilder, config: &SvdConfig, device: Device, dtype: DType) -> Result<Self> {
        let unet = UNetSpatioTemporalConditionModel::new(vb.pp("unet"), &config.unet)?;
        let vae = AutoencoderKLTemporalDecoder::new(vb.pp("vae"), &config.vae)?;
        let image_encoder =
            ClipVisionModelWithProjection::new(vb.pp("image_encoder"), &config.clip)?;
        let scheduler = EulerDiscreteScheduler::new(config.scheduler.clone());

        Ok(Self {
            unet,
            vae,
            image_encoder,
            scheduler,
            device,
            dtype,
        })
    }

    /// Create a pipeline from separate VarBuilders for each component (no prefix needed)
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

        Ok(Self {
            unet,
            vae,
            image_encoder,
            scheduler,
            device,
            dtype,
        })
    }

    /// Generate video frames from an input image
    pub fn generate(&mut self, image: &Tensor, config: &SvdInferenceConfig) -> Result<Tensor> {
        let batch_size = 1;
        let num_frames = config.num_frames;
        let height = config.height;
        let width = config.width;
        let latent_height = height / 8;
        let latent_width = width / 8;

        // 1. Encode input image with CLIP (requires 224x224)
        // Use bilinear for smoother interpolation (diffusers uses antialiased resize)
        let clip_image = image.interpolate2d(224, 224)?;
        let image_normalized = normalize_for_clip(&clip_image, &self.device)?;
        let image_embeddings = self.image_encoder.forward(&image_normalized)?;

        // Repeat for all frames: [B, D] -> [B*F, 1, D]
        let embed_dim = image_embeddings.dim(1)?;
        let image_embeddings = image_embeddings
            .unsqueeze(1)? // [B, 1, D]
            .repeat((1, num_frames, 1))? // [B, F, D]
            .reshape((batch_size * num_frames, 1, embed_dim))?; // [B*F, 1, D]

        // 2. Encode input image to latent space with VAE
        // NOTE: diffusers adds noise in pixel space BEFORE encoding
        // See: pipeline_stable_video_diffusion.py:511-512
        let noise_aug_strength = config.noise_aug_strength;
        let noise = Tensor::randn_like(image, 0.0, 1.0)?;
        let image_augmented = (image + &(noise * noise_aug_strength)?)?;
        let image_latents = self.vae.encode_to_latent(&image_augmented)?;

        // Repeat image latents for all frames for conditioning [B*F, 4, H, W]
        let image_cond_latents = image_latents
            .unsqueeze(1)? // [B, 1, 4, H, W]
            .repeat((1, num_frames, 1, 1, 1))? // [B, F, 4, H, W]
            .reshape((batch_size * num_frames, 4, latent_height, latent_width))?;

        // Create noisy latents: start from noise
        let latents = Tensor::randn(
            0f32,
            1f32,
            (batch_size * num_frames, 4, latent_height, latent_width),
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        // 3. Prepare added time IDs
        // NOTE: SVD was conditioned on fps-1 during training
        // See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
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

        // 4. Set up scheduler
        self.scheduler
            .set_timesteps(config.num_inference_steps, &self.device)?;
        let timesteps: Vec<f64> = self.scheduler.timesteps().to_vec();

        // Scale initial noise
        let mut latents = (latents * self.scheduler.init_noise_sigma())?;

        // 5. Prepare per-frame guidance scale (as in diffusers)
        // Guidance interpolates from min to max across FRAMES, not steps
        // Shape: [B*F] -> will be reshaped for broadcasting
        let guidance_scales: Vec<f32> = (0..num_frames)
            .map(|f| {
                let t = if num_frames > 1 {
                    f as f64 / (num_frames - 1) as f64
                } else {
                    0.0
                };
                (config.min_guidance_scale + (config.max_guidance_scale - config.min_guidance_scale) * t) as f32
            })
            .collect();
        
        // Repeat for batch_size and create tensor [B*F, 1, 1, 1]
        let guidance_scale_vec: Vec<f32> = (0..batch_size)
            .flat_map(|_| guidance_scales.iter().copied())
            .collect();
        let guidance_scale_tensor = Tensor::new(guidance_scale_vec.as_slice(), &self.device)?
            .to_dtype(self.dtype)?
            .reshape((batch_size * num_frames, 1, 1, 1))?;

        // Check if we need CFG (any guidance > 1.0)
        let do_classifier_free_guidance = config.max_guidance_scale > 1.0;

        // 6. Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Expand latents for classifier-free guidance if needed
            let latent_model_input = self.scheduler.scale_model_input(&latents, i)?;

            // Create timestep tensor
            let timestep = Tensor::new(&[t], &self.device)?
                .to_dtype(self.dtype)?
                .repeat(batch_size * num_frames)?;

            // Concatenate noise latents with image conditioning latents -> 8 channels
            let noise_pred = if do_classifier_free_guidance {
                // Prepare uncond inputs (zeros)
                let zeros_cond_latents = image_cond_latents.zeros_like()?;
                let latent_input_uncond = Tensor::cat(&[&latent_model_input, &zeros_cond_latents], 1)?;
                let zeros_image_embeddings = image_embeddings.zeros_like()?;

                // Uncond forward pass
                let noise_pred_uncond = self.unet.forward(
                    &latent_input_uncond,
                    &timestep,
                    &zeros_image_embeddings,
                    &added_time_ids,
                    num_frames,
                    None,
                )?;

                // Cond forward pass
                let latent_input_cond = Tensor::cat(&[&latent_model_input, &image_cond_latents], 1)?;
                let noise_pred_cond = self.unet.forward(
                    &latent_input_cond,
                    &timestep,
                    &image_embeddings,
                    &added_time_ids,
                    num_frames,
                    None,
                )?;

                // Combine with per-frame guidance: uncond + scale * (cond - uncond)
                let diff = (&noise_pred_cond - &noise_pred_uncond)?;
                (&noise_pred_uncond + diff.broadcast_mul(&guidance_scale_tensor)?)?
            } else {
                let latent_input = Tensor::cat(&[&latent_model_input, &image_cond_latents], 1)?;
                self.unet.forward(
                    &latent_input,
                    &timestep,
                    &image_embeddings,
                    &added_time_ids,
                    num_frames,
                    None,
                )?
            };

            // Scheduler step
            let output = self.scheduler.step(&noise_pred, i, &latents)?;
            latents = output.prev_sample;
        }

        // 7. Decode latents to video frames
        let video_frames = self.vae.decode(&latents, num_frames, config.decode_chunk_size)?;

        // Reshape to [B, F, C, H, W]
        let video_frames = video_frames.reshape((batch_size, num_frames, 3, height, width))?;

        // Denormalize: [-1, 1] -> [0, 1]
        let video_frames = ((video_frames + 1.0)? / 2.0)?;
        let video_frames = video_frames.clamp(0.0, 1.0)?;

        Ok(video_frames)
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Load image from file and preprocess for SVD
pub fn load_image(
    path: &str,
    height: usize,
    width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    use std::io::Read;

    // Read image file
    let mut file = std::fs::File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open image: {}", e)))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read image: {}", e)))?;

    // Decode image
    let img = image::load_from_memory(&buffer)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to decode image: {}", e)))?;

    // Resize to target dimensions
    let img = img.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let img = img.to_rgb8();

    // Convert to tensor [1, 3, H, W] normalized to [-1, 1]
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

/// Save video frames to files
pub fn save_video_frames(frames: &Tensor, output_dir: &str) -> Result<()> {
    use std::fs;

    fs::create_dir_all(output_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create output dir: {}", e)))?;

    let (batch, num_frames, _channels, height, width) = frames.dims5()?;

    for b in 0..batch {
        for f in 0..num_frames {
            let frame = frames.i((b, f, .., .., ..))?;
            let frame = (frame * 255.0)?.to_dtype(DType::U8)?;

            // Convert to image buffer
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
