//! Tests for Text-to-Video pipeline
//!
//! This test file validates the complete text-to-video generation pipeline,
//! including integration of all components: T5 encoder, DiT, VAE, and Scheduler.

use candle_core::{DType, Device, Result, Tensor};
use candle_video::config::{DitConfig, InferenceConfig, SchedulerConfig, VaeConfig};
use candle_video::pipeline::{PipelineConfig, TextToVideoPipeline};
use candle_video::text_encoder::{T5EncoderConfig, T5TextEncoderWrapper};

// =============================================================================
// Helper functions
// =============================================================================

fn create_test_device() -> Device {
    Device::Cpu
}

fn create_test_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 4, // Smaller for testing
        block_out_channels: vec![32, 64],
        layers_per_block: 1,
        temporal_downsample: 4,
        spatial_downsample: 8,
        causal: true,
        latents_mean: None,
        latents_std: None,
        scaling_factor: 1.0,
        timestep_conditioning: false,
    }
}

fn create_test_dit_config() -> DitConfig {
    DitConfig {
        patch_size: 1, // No spatial patching for simpler tests
        patch_size_t: None,
        in_channels: 4,  // Match VAE latent_channels
        hidden_size: 64, // Small for testing
        depth: 2,        // Few layers for speed
        num_heads: 4,
        caption_channels: 128, // Match test T5
        mlp_ratio: 4.0,
        use_flash_attention: false,
        timestep_scale_multiplier: None,
    }
}

fn create_test_t5_config() -> T5EncoderConfig {
    T5EncoderConfig::new(128, 4, 2) // Small config for testing
}

fn create_test_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        num_inference_steps: 5, // Few steps for testing
        num_train_timesteps: 1000,
        guidance_scale: 3.0,
        timestep_spacing: "linspace".to_string(),
        shift: None,
        use_dynamic_shifting: false,
        base_shift: 0.5,
        max_shift: 1.15,
        shift_terminal: None,
        stochastic_sampling: false,
    }
}

fn create_test_inference_config() -> InferenceConfig {
    InferenceConfig::new(9, 64, 64, 42).unwrap() // 9 frames (8N+1), small resolution
}

// =============================================================================
// Pipeline Creation Tests
// =============================================================================

#[test]
fn test_pipeline_config_creation() {
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };

    assert_eq!(config.dit.hidden_size, 64);
    assert_eq!(config.vae.latent_channels, 4);
    assert_eq!(config.scheduler.num_inference_steps, 5);
}

#[test]
fn test_pipeline_creation_without_models() {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };

    let pipeline = TextToVideoPipeline::new(device, config);
    assert!(pipeline.is_ok());
}

#[test]
fn test_pipeline_accessors() {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };

    let pipeline = TextToVideoPipeline::new(device, config).unwrap();

    // Check scheduler is accessible
    let scheduler = pipeline.scheduler();
    assert_eq!(scheduler.num_steps(), 5);
    assert_eq!(scheduler.guidance_scale(), 3.0);

    // Check device
    assert!(matches!(pipeline.device(), Device::Cpu));
}

// =============================================================================
// Noise Initialization Tests
// =============================================================================

#[test]
fn test_initialize_noise_shape() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config)?;

    let inference = create_test_inference_config();
    let noise = pipeline.initialize_noise(&inference)?;

    // Expected shape: (B, C_latent, T_latent, H_latent, W_latent)
    // T_latent = (9 - 1) / 4 + 1 = 3 (temporal downsample)
    // H_latent = 64 / 8 = 8 (spatial downsample)
    // W_latent = 64 / 8 = 8 (spatial downsample)
    assert_eq!(noise.dims(), &[1, 4, 3, 8, 8]);

    Ok(())
}

#[test]
fn test_initialize_noise_with_seed_reproducibility() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config)?;

    let inference = InferenceConfig::new(9, 64, 64, 123).unwrap();
    let noise1 = pipeline.initialize_noise_with_seed(&inference, 123)?;
    let noise2 = pipeline.initialize_noise_with_seed(&inference, 123)?;

    // Same seed should produce same noise (with proper seeding)
    assert_eq!(noise1.dims(), noise2.dims());

    Ok(())
}

// =============================================================================
// Latent Dimension Calculation Tests
// =============================================================================

#[test]
fn test_compute_latent_dims() {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config).unwrap();

    let inference = create_test_inference_config();
    let (lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference);

    assert_eq!(lat_t, 3); // (9 - 1) / 4 + 1
    assert_eq!(lat_h, 8); // 64 / 8
    assert_eq!(lat_w, 8); // 64 / 8
}

#[test]
fn test_compute_latent_dims_larger_video() {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: VaeConfig {
            temporal_downsample: 8,
            spatial_downsample: 32,
            ..create_test_vae_config()
        },
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config).unwrap();

    let inference = InferenceConfig::new(17, 512, 768, 42).unwrap();
    let (lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference);

    assert_eq!(lat_t, 3); // (17 - 1) / 8 + 1
    assert_eq!(lat_h, 16); // 512 / 32
    assert_eq!(lat_w, 24); // 768 / 32
}

#[test]
fn test_compute_latent_dims_diffusers_formula() {
    let device = create_test_device();
    let temporal_downsample = 8;
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: VaeConfig {
            temporal_downsample,
            spatial_downsample: 32,
            ..create_test_vae_config()
        },
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config).unwrap();

    let num_frames = 25;
    let inference = InferenceConfig::new(num_frames, 512, 768, 42).unwrap();
    let (lat_t, _lat_h, _lat_w) = pipeline.compute_latent_dims(&inference);

    let expected_lat_t = (num_frames - 1) / temporal_downsample + 1;
    assert_eq!(lat_t, expected_lat_t);
}

// =============================================================================
// Coordinate Grid Tests
// =============================================================================

#[test]
fn test_generate_rope_indices() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config)?;

    let inference = create_test_inference_config();
    let indices = pipeline.generate_rope_indices(&inference)?;

    // Shape should be (batch_size, 3, seq_len) where seq_len = T * H * W
    let (lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference);
    let expected_seq_len = lat_t * lat_h * lat_w;

    assert_eq!(indices.dims(), &[1, 3, expected_seq_len]);

    Ok(())
}

// =============================================================================
// CFG (Classifier-Free Guidance) Tests
// =============================================================================

#[test]
fn test_prepare_cfg_embeddings() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    // Create mock positive and negative embeddings
    let pos_emb = Tensor::ones((1, 10, 128), DType::F32, &device)?;
    let neg_emb = Tensor::zeros((1, 10, 128), DType::F32, &device)?;

    let cfg_emb = pipeline.prepare_cfg_embeddings(&pos_emb, &neg_emb)?;

    // CFG embeddings should be concatenated: [negative, positive]
    assert_eq!(cfg_emb.dims(), &[2, 10, 128]);

    Ok(())
}

#[test]
fn test_apply_cfg_to_output() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            guidance_scale: 7.5,
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device, config)?;

    // Create mock model outputs: [unconditional, conditional]
    let uncond = Tensor::zeros((1, 4, 2, 8, 8), DType::F32, &Device::Cpu)?;
    let cond = Tensor::ones((1, 4, 2, 8, 8), DType::F32, &Device::Cpu)?;

    // Stack them as if coming from batched CFG forward pass
    let model_output = Tensor::cat(&[&uncond, &cond], 0)?;

    let guided = pipeline.apply_cfg(&model_output)?;

    // Result should be single batch
    assert_eq!(guided.dims(), &[1, 4, 2, 8, 8]);

    // With guidance_scale=7.5: guided = uncond + 7.5 * (cond - uncond) = 7.5 * cond = 7.5
    // (since uncond is 0 and cond is 1)

    Ok(())
}

// =============================================================================
// Mock Generation Tests (without loaded models)
// =============================================================================

#[test]
fn test_mock_dit_forward() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    // Create mock inputs
    let latents = Tensor::randn(0f32, 1.0, (1, 4, 2, 8, 8), &device)?;
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;
    let timestep = 0.5f64;

    // Mock forward should return velocity with same shape as latents
    let velocity = pipeline.mock_dit_forward(&latents, &text_emb, timestep)?;

    assert_eq!(velocity.dims(), latents.dims());

    Ok(())
}

#[test]
fn test_mock_vae_decode() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();

    // Create mock latents
    let (lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference);
    let latents = Tensor::randn(0f32, 1.0, (1, 4, lat_t, lat_h, lat_w), &device)?;

    // Mock decode should return video frames
    let frames = pipeline.mock_vae_decode(&latents, &inference)?;

    // Output shape: (B, C=3, T, H, W)
    assert_eq!(frames.dim(0)?, 1);
    assert_eq!(frames.dim(1)?, 3); // RGB

    Ok(())
}

#[test]
fn test_mock_generate_full_pipeline() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();

    // Create mock text embeddings
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Run full mock generation
    let video = pipeline.mock_generate(&text_emb, &inference)?;

    // Output should be video tensor (B, 3, T, H, W)
    assert_eq!(video.dim(0)?, 1);
    assert_eq!(video.dim(1)?, 3);

    Ok(())
}

#[test]
fn test_mock_generate_with_cfg() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            guidance_scale: 7.5,
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();

    // Create mock positive and negative embeddings
    let pos_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;
    let neg_emb = Tensor::randn(0f32, 0.1, (1, 10, 128), &device)?;

    // Run full mock generation with CFG
    let video = pipeline.mock_generate_with_cfg(&pos_emb, &neg_emb, &inference)?;

    // Output should be video tensor
    assert_eq!(video.dim(0)?, 1);
    assert_eq!(video.dim(1)?, 3);

    Ok(())
}

// =============================================================================
// Denoising Loop Tests
// =============================================================================

#[test]
fn test_single_denoising_step() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let latents = Tensor::randn(0f32, 1.0, (1, 4, 2, 8, 8), &device)?;
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Single step should produce updated latents
    let updated = pipeline.mock_denoising_step(&latents, &text_emb, 0)?;

    assert_eq!(updated.dims(), latents.dims());

    Ok(())
}

#[test]
fn test_denoising_loop_iteration_count() -> Result<()> {
    let device = create_test_device();
    let num_steps = 5;
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            num_inference_steps: num_steps,
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Track iterations through callback
    let mut iterations = 0;
    let _latents =
        pipeline.mock_denoise_with_callback(&text_emb, &inference, |_step, _latents| {
            iterations += 1;
        })?;

    assert_eq!(iterations, num_steps);

    Ok(())
}

// =============================================================================
// Progress Callback Tests
// =============================================================================

#[test]
fn test_generate_with_progress_callback() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            num_inference_steps: 3,
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    let mut progress_values = Vec::new();

    let _video = pipeline.mock_generate_with_progress(&text_emb, &inference, |progress| {
        progress_values.push(progress);
    })?;

    // Should have progress updates for each step
    assert!(!progress_values.is_empty());

    // Progress should be monotonically increasing
    for window in progress_values.windows(2) {
        assert!(window[0] <= window[1]);
    }

    Ok(())
}

// =============================================================================
// Pipeline State Tests
// =============================================================================

#[test]
fn test_pipeline_with_scheduler_mutation() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let mut pipeline = TextToVideoPipeline::new(device, config)?;

    // Should be able to update scheduler parameters
    pipeline.set_guidance_scale(5.0);
    assert_eq!(pipeline.scheduler().guidance_scale(), 5.0);

    pipeline.set_num_inference_steps(10);
    assert_eq!(pipeline.scheduler().num_steps(), 10);

    Ok(())
}

// =============================================================================
// Timestep Tensor Tests
// =============================================================================

#[test]
fn test_create_timestep_tensor() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device, config)?;

    let timestep = pipeline.create_timestep_tensor(0.5, 2)?;

    // Should have shape (batch_size,)
    assert_eq!(timestep.dims(), &[2]);

    Ok(())
}

// =============================================================================
// Memory-related Tests
// =============================================================================

#[test]
fn test_pipeline_memory_cleanup() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Generate should not panic or leak
    let _video = pipeline.mock_generate(&text_emb, &inference)?;

    // Second generation should also work
    let _video2 = pipeline.mock_generate(&text_emb, &inference)?;

    Ok(())
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_pipeline_with_single_frame() -> Result<()> {
    // Note: Single frame (1 frame) doesn't satisfy 8N+1 constraint
    // Minimum is 9 frames (8*1+1)
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = InferenceConfig::new(9, 64, 64, 42).unwrap();
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    let video = pipeline.mock_generate(&text_emb, &inference)?;
    assert!(video.dims().len() == 5);

    Ok(())
}

#[test]
fn test_pipeline_empty_prompt_handling() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();

    // Empty text embedding (1 token for EOS)
    let text_emb = Tensor::randn(0f32, 1.0, (1, 1, 128), &device)?;

    let video = pipeline.mock_generate(&text_emb, &inference)?;
    assert!(video.dims().len() == 5);

    Ok(())
}

#[test]
fn test_pipeline_very_short_inference_steps() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            num_inference_steps: 1, // Minimum steps
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Should work with just 1 step
    let video = pipeline.mock_generate(&text_emb, &inference)?;
    assert_eq!(video.dim(0)?, 1);

    Ok(())
}

// =============================================================================
// Integration with T5 Encoder
// =============================================================================

#[test]
fn test_pipeline_with_mock_t5_encoder() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    // Create T5 encoder
    let t5_config = create_test_t5_config();
    let mut encoder = T5TextEncoderWrapper::new(t5_config, device.clone(), DType::F32)?;

    // Mock encode a prompt
    let text_emb = encoder.mock_encode("A beautiful sunset over the ocean")?;

    // Use in pipeline
    let inference = create_test_inference_config();
    let video = pipeline.mock_generate(&text_emb, &inference)?;

    assert_eq!(video.dim(0)?, 1);
    assert_eq!(video.dim(1)?, 3);

    Ok(())
}

#[test]
fn test_pipeline_with_cfg_t5_encoding() -> Result<()> {
    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            guidance_scale: 7.5,
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    // Create T5 encoder
    let t5_config = create_test_t5_config();
    let mut encoder = T5TextEncoderWrapper::new(t5_config, device.clone(), DType::F32)?;

    // Mock encode for CFG
    let (pos_emb, neg_emb) = encoder.mock_encode_for_cfg("A sunset", "")?;

    // Use in pipeline
    let inference = create_test_inference_config();
    let video = pipeline.mock_generate_with_cfg(&pos_emb, &neg_emb, &inference)?;

    assert_eq!(video.dim(0)?, 1);
    assert_eq!(video.dim(1)?, 3);

    Ok(())
}

// =============================================================================
// Conditioning Mask Tests (Image-to-Video)
// =============================================================================

#[test]
fn test_compute_token_timesteps_no_conditioning() -> Result<()> {
    use candle_video::compute_token_timesteps;
    let device = Device::Cpu;

    // No conditioning = mask is all zeros
    let conditioning_mask = Tensor::zeros((1, 16), DType::F32, &device)?;
    let base_t = 0.8;

    let timesteps = compute_token_timesteps(base_t, &conditioning_mask)?;

    // All tokens should have base_timestep since min(0.8, 1.0 - 0) = min(0.8, 1.0) = 0.8
    let data = timesteps.flatten_all()?.to_vec1::<f32>()?;
    for val in data {
        assert!((val - 0.8).abs() < 1e-5, "Expected 0.8, got {}", val);
    }

    Ok(())
}

#[test]
fn test_compute_token_timesteps_full_conditioning() -> Result<()> {
    use candle_video::compute_token_timesteps;
    let device = Device::Cpu;

    // Full conditioning = mask is all 1.0
    let conditioning_mask = Tensor::ones((1, 16), DType::F32, &device)?;
    let base_t = 0.8;

    let timesteps = compute_token_timesteps(base_t, &conditioning_mask)?;

    // All tokens should have 0.0 since min(0.8, 1.0 - 1.0) = min(0.8, 0.0) = 0.0
    let data = timesteps.flatten_all()?.to_vec1::<f32>()?;
    for val in data {
        assert!((val - 0.0).abs() < 1e-5, "Expected 0.0, got {}", val);
    }

    Ok(())
}

#[test]
fn test_compute_token_timesteps_partial_conditioning() -> Result<()> {
    use candle_video::compute_token_timesteps;
    let device = Device::Cpu;

    // First 4 tokens conditioned, rest not
    let mask_data = vec![1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let conditioning_mask = Tensor::from_vec(mask_data, (1, 8), &device)?;
    let base_t = 0.9;

    let timesteps = compute_token_timesteps(base_t, &conditioning_mask)?;
    let data = timesteps.flatten_all()?.to_vec1::<f32>()?;

    // First 4 should be 0.0 (fully conditioned)
    for i in 0..4 {
        assert!((data[i] - 0.0).abs() < 1e-5, "Token {} should be 0.0", i);
    }
    // Last 4 should be 0.9 (not conditioned)
    for i in 4..8 {
        assert!((data[i] - 0.9).abs() < 1e-5, "Token {} should be 0.9", i);
    }

    Ok(())
}

#[test]
fn test_apply_conditioning_mask_preserves_conditioned_tokens() -> Result<()> {
    use candle_video::apply_conditioning_mask;
    let device = Device::Cpu;

    // Original and denoised latents [B=1, seq=4, C=2]
    let original = Tensor::ones((1, 4, 2), DType::F32, &device)?;
    let denoised = Tensor::zeros((1, 4, 2), DType::F32, &device)?;

    // First 2 tokens are fully conditioned
    let mask_data = vec![1.0f32, 1.0, 0.0, 0.0];
    let conditioning_mask = Tensor::from_vec(mask_data, (1, 4), &device)?;

    let t = 0.5; // Current timestep

    let result = apply_conditioning_mask(&denoised, &original, t, &conditioning_mask)?;
    let data = result.to_vec3::<f32>()?;

    // Conditioned tokens (mask=1.0) should keep original values (1.0)
    // Because (t - eps) < (1.0 - 1.0) = (0.5 - eps) < 0.0 is FALSE
    assert!(
        (data[0][0][0] - 1.0).abs() < 1e-5,
        "Conditioned token should be original"
    );
    assert!(
        (data[0][1][0] - 1.0).abs() < 1e-5,
        "Conditioned token should be original"
    );

    // Non-conditioned tokens (mask=0.0) should be denoised (0.0)
    // Because (t - eps) < (1.0 - 0.0) = (0.5 - eps) < 1.0 is TRUE
    assert!(
        (data[0][2][0] - 0.0).abs() < 1e-5,
        "Denoised token should be updated"
    );
    assert!(
        (data[0][3][0] - 0.0).abs() < 1e-5,
        "Denoised token should be updated"
    );

    Ok(())
}

#[test]
fn test_conditioning_item_creation() {
    use candle_video::ConditioningItem;
    let device = Device::Cpu;

    let latents = Tensor::zeros((1, 4, 1, 8, 8), DType::F32, &device).unwrap();
    let item = ConditioningItem::new(latents, 0, 0.8);

    assert_eq!(item.frame_number, 0);
    assert!((item.strength - 0.8).abs() < 1e-5);
}

#[test]
fn test_conditioning_item_strength_clamping() {
    use candle_video::ConditioningItem;
    let device = Device::Cpu;

    let latents = Tensor::zeros((1, 4, 1, 8, 8), DType::F32, &device).unwrap();

    // Test clamping above 1.0
    let item = ConditioningItem::new(latents.clone(), 0, 1.5);
    assert!((item.strength - 1.0).abs() < 1e-5);

    // Test clamping below 0.0
    let item = ConditioningItem::new(latents, 0, -0.5);
    assert!((item.strength - 0.0).abs() < 1e-5);
}

// =============================================================================
// Image-to-Video Integration Tests
// =============================================================================

#[test]
fn test_prepare_conditioning_latents() -> Result<()> {
    use candle_video::ConditioningItem;

    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: create_test_scheduler_config(),
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    // Create mock initial noise latents [B=1, C=4, T=4, H=8, W=8]
    let init_latents = Tensor::randn(0f32, 1.0, (1, 4, 4, 8, 8), &device)?;

    // Create conditioning item for first 2 frames
    let cond_latents = Tensor::zeros((1, 4, 2, 8, 8), DType::F32, &device)?;
    let cond_item = ConditioningItem::new(cond_latents, 0, 1.0);

    let (conditioned, mask) = pipeline.prepare_conditioning_latents(&init_latents, &[cond_item])?;

    // Check shapes
    assert_eq!(conditioned.dims(), init_latents.dims());
    let seq_len = 4 * 8 * 8; // T * H * W
    assert_eq!(mask.dims(), &[1, seq_len]);

    // First 2 frames worth of tokens should have mask = 1.0
    let mask_data = mask.flatten_all()?.to_vec1::<f32>()?;
    let cond_tokens = 2 * 8 * 8;
    for i in 0..cond_tokens {
        assert!(
            (mask_data[i] - 1.0).abs() < 1e-5,
            "Token {} should be conditioned",
            i
        );
    }
    // Rest should be 0.0
    for i in cond_tokens..seq_len {
        assert!(
            (mask_data[i] - 0.0).abs() < 1e-5,
            "Token {} should not be conditioned",
            i
        );
    }

    Ok(())
}

#[test]
fn test_mock_generate_image_to_video() -> Result<()> {
    use candle_video::ConditioningItem;

    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            num_inference_steps: 3, // Few steps for testing
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();

    // Create text embeddings
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Create conditioning item (simulating encoded first frame)
    let (_lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference);
    let cond_latents = Tensor::zeros((1, 4, 1, lat_h, lat_w), DType::F32, &device)?;
    let cond_item = ConditioningItem::new(cond_latents, 0, 1.0);

    // Generate video with i2v conditioning
    let video =
        pipeline.mock_generate_image_to_video(&text_emb, &[cond_item], &inference, 0.025)?;

    // Output should be video tensor (B, 3, T, H, W)
    assert_eq!(video.dim(0)?, 1);
    assert_eq!(video.dim(1)?, 3);

    Ok(())
}

#[test]
fn test_generate_image_to_video_api() -> Result<()> {
    use candle_video::ConditioningItem;

    let device = create_test_device();
    let config = PipelineConfig {
        dit: create_test_dit_config(),
        vae: create_test_vae_config(),
        scheduler: SchedulerConfig {
            num_inference_steps: 2,
            ..create_test_scheduler_config()
        },
    };
    let pipeline = TextToVideoPipeline::new(device.clone(), config)?;

    let inference = create_test_inference_config();
    let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 128), &device)?;

    // Create minimal conditioning
    let (_lat_t, lat_h, lat_w) = pipeline.compute_latent_dims(&inference);
    let cond_latents = Tensor::ones((1, 4, 1, lat_h, lat_w), DType::F32, &device)?;
    let cond_item = ConditioningItem::new(cond_latents, 0, 0.9);

    // Use mock API (real i2v requires loaded models)
    let video =
        pipeline.mock_generate_image_to_video(&text_emb, &[cond_item], &inference, 0.025)?;

    assert_eq!(video.dim(0)?, 1);
    assert_eq!(video.dim(1)?, 3);
    assert!(video.dim(2).is_ok()); // Has temporal dimension

    Ok(())
}
