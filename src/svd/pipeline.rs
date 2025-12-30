//! SVD Pipeline - Image-to-Video Generation
//!
//! Main pipeline for Stable Video Diffusion inference.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::{debug, info};

use crate::svd::{
    AutoencoderKLTemporalDecoder, ClipVisionModelWithProjection, EulerDiscreteScheduler, SvdConfig,
    SvdInferenceConfig, UNetSpatioTemporalConditionModel, normalize_for_clip,
};

/// Dump tensor to .npy file for comparison with Python reference.
/// Only active when DUMP_TENSORS env var is set.
#[allow(dead_code)]
fn dump_tensor(name: &str, tensor: &Tensor) {
    if std::env::var("DUMP_TENSORS").is_ok() {
        let dir = std::path::Path::new("output/rust_tensors");
        std::fs::create_dir_all(dir).ok();
        
        // Convert to f32 and save as raw binary
        if let Ok(t) = tensor.to_dtype(DType::F32)
            && let Ok(flat) = t.flatten_all()
                && let Ok(data) = flat.to_vec1::<f32>() {
                    // Save shape
                    let shape: Vec<usize> = t.dims().to_vec();
                    let shape_path = dir.join(format!("{}.shape", name));
                    let shape_str = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
                    std::fs::write(&shape_path, shape_str).ok();
                    
                    // Save raw f32 data
                    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(dir.join(format!("{}.bin", name)), bytes).ok();
                    debug!("Dumped tensor {} shape={:?}", name, shape);
                }
    }
}

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
    ///
    /// # Latent Shape Convention
    /// This implementation uses `[B*F, C, H, W]` (4D) tensors with `num_frames` passed explicitly.
    /// This differs from diffusers which uses `[B, F, C, H, W]` (5D) at API level.
    /// Both are functionally equivalent - diffusers also flattens to 4D internally for 2D convolutions.
    /// The `num_frames` parameter enables proper temporal reshaping in SpatioTemporalResBlock.
    pub fn generate(&mut self, image: &Tensor, config: &SvdInferenceConfig) -> Result<Tensor> {
        let batch_size = 1;
        let num_frames = config.num_frames;
        let height = config.height;
        let width = config.width;
        let latent_height = height / 8;
        let latent_width = width / 8;

        info!(
            num_frames,
            height, width, latent_height, latent_width, "SVD generate start"
        );
        debug!(input_shape = ?image.dims(), dtype = ?image.dtype(), "Input image");

        // 1. Encode input image with CLIP (requires 224x224)
        // Use bilinear for smoother interpolation (diffusers uses antialiased resize)
        let clip_image = image.interpolate2d(224, 224)?;
        // Input image is in [-1, 1], but CLIP expects [0, 1] for normalization
        let clip_image_01 = ((clip_image + 1.0)? / 2.0)?;
        let image_normalized = normalize_for_clip(&clip_image_01, &self.device)?;
        let image_embeddings = self.image_encoder.forward(&image_normalized)?;
        // NOTE on shape convention:
        // Diffusers uses [B, 1, D] at API level and expands internally in UNet.
        // Our UNet expects pre-expanded [B*F, 1, D] because it processes 4D [B*F, C, H, W] latents.
        // This is functionally equivalent - just different layer of abstraction.
        let embed_dim = image_embeddings.dim(1)?;
        let image_embeddings = image_embeddings
            .unsqueeze(1)?                                    // [B, 1, D]
            .repeat((1, num_frames, 1))?                      // [B, F, D]
            .reshape((batch_size * num_frames, 1, embed_dim))?; // [B*F, 1, D]
        dump_tensor("image_embeddings_raw", &image_embeddings);

        debug!(image_embeddings_shape = ?image_embeddings.dims(), "CLIP image embeddings");

        // 2. Encode input image to latent space with VAE
        // NOTE: diffusers adds noise in pixel space BEFORE encoding
        // See: pipeline_stable_video_diffusion.py:511-512
        let noise_aug_strength = config.noise_aug_strength;
        let noise = Tensor::randn_like(image, 0.0, 1.0)?;
        let image_augmented = (image + &(noise * noise_aug_strength)?)?;
        dump_tensor("vae_input", &image_augmented);
        let image_latents = self.vae.encode_to_latent(&image_augmented)?;
        dump_tensor("image_latents_raw", &image_latents);

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
        // Our UNet expects [B*F, 3] because it processes 4D latents
        let added_time_ids = Tensor::new(
            &[[
                config.fps.saturating_sub(1) as f32,
                config.motion_bucket_id as f32,
                noise_aug_strength as f32,
            ]],
            &self.device,
        )?
        .to_dtype(self.dtype)?
        .repeat((batch_size * num_frames, 1))?;  // [B*F, 3]
        dump_tensor("added_time_ids", &added_time_ids);

        // 4. Set up scheduler
        self.scheduler
            .set_timesteps(config.num_inference_steps, &self.device)?;
        let timesteps: Vec<f64> = self.scheduler.timesteps().to_vec();

        debug!(latents_shape = ?latents.dims(), image_cond_latents_shape = ?image_cond_latents.dims(), "Initial tensors");

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
                (config.min_guidance_scale
                    + (config.max_guidance_scale - config.min_guidance_scale) * t)
                    as f32
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

        // Pre-allocate CFG tensors to avoid repeated allocation in loop
        let (image_cond_latents_cfg, encoder_states_cfg, added_time_ids_cfg) =
            if do_classifier_free_guidance {
                // [zeros, real] for unconditional/conditional
                let zeros_cond = image_cond_latents.zeros_like()?;
                let cond_cfg = Tensor::cat(&[&zeros_cond, &image_cond_latents], 0)?;
                drop(zeros_cond); // Explicit drop to free memory

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

        // 6. Denoising loop
        let total_steps = timesteps.len();
        for (i, &t) in timesteps.iter().enumerate() {
            println!("  Step {}/{} (t={:.4})", i + 1, total_steps, t);
            debug!(step = i, timestep = t, latents_shape = ?latents.dims(), "Denoising step");
            // Create timestep tensor - will be expanded for CFG if needed
            let base_timestep = Tensor::new(&[t], &self.device)?
                .to_dtype(self.dtype)?
                .repeat(batch_size * num_frames)?;

            // Concatenate noise latents with image conditioning latents -> 8 channels
            let noise_pred = if do_classifier_free_guidance {
                // === BATCHED CFG: single forward pass with doubled batch ===
                // Follows diffusers: cat([latents]*2) -> scale_model_input -> cat(..., image_latents)

                // 1. Expand latents for CFG: [B*F, C, H, W] -> [2*B*F, C, H, W]
                let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

                // 2. Scale AFTER expansion (diffusers order)
                let latent_model_input =
                    self.scheduler.scale_model_input(&latent_model_input, i)?;

                // 3. Concatenate along channel dimension (using pre-allocated image_cond_latents_cfg)
                let latent_input = Tensor::cat(&[&latent_model_input, &image_cond_latents_cfg], 1)?;

                // 4. Expand timestep for doubled batch
                let timestep_cfg = Tensor::cat(&[&base_timestep, &base_timestep], 0)?;

                // 5. Single UNet forward pass (using pre-allocated encoder_states_cfg and added_time_ids_cfg)
                let noise_pred = self.unet.forward(
                    &latent_input,
                    &timestep_cfg,
                    &encoder_states_cfg,
                    &added_time_ids_cfg,
                    num_frames,
                    None,
                )?;

                // 8. Split predictions: chunk(2) along batch dimension
                let half_batch = batch_size * num_frames;
                let noise_pred_uncond = noise_pred.narrow(0, 0, half_batch)?;
                let noise_pred_cond = noise_pred.narrow(0, half_batch, half_batch)?;

                // 9. Apply per-frame guidance: uncond + scale * (cond - uncond)
                let diff = (&noise_pred_cond - &noise_pred_uncond)?;
                (&noise_pred_uncond + diff.broadcast_mul(&guidance_scale_tensor)?)?
            } else {
                // No CFG - single forward pass
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

            // Scheduler step
            debug!(noise_pred_shape = ?noise_pred.dims(), "Before scheduler step");
            
            // Debug: print tensor statistics to identify where values become zero
            if let Ok(np_f32) = noise_pred.to_dtype(candle_core::DType::F32) {
                if let Ok(flat) = np_f32.flatten_all() {
                    let min = flat.min(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                    let max = flat.max(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                    println!("    noise_pred: min={:?}, max={:?}", min, max);
                }
            }
            
            let output = self.scheduler.step(&noise_pred, i, &latents)?;
            latents = output.prev_sample;
            
            // Debug: latents after scheduler step
            if let Ok(lat_f32) = latents.to_dtype(candle_core::DType::F32) {
                if let Ok(flat) = lat_f32.flatten_all() {
                    let min = flat.min(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                    let max = flat.max(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
                    println!("    latents: min={:?}, max={:?}", min, max);
                }
            }
        }

        info!("Denoising complete, decoding latents");
        debug!(final_latents_shape = ?latents.dims(), "Latents before VAE decode");

        // 7. Decode latents to video frames
        let video_frames = self
            .vae
            .decode(&latents, num_frames, config.decode_chunk_size)?;
        debug!(video_frames_shape = ?video_frames.dims(), "After VAE decode");

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
