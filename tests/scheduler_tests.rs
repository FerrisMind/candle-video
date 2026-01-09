//! Comprehensive tests for Rectified Flow Scheduler
//!
//! Following TDD methodology, these tests cover:
//! - Timestep generation (linspace, linear_quadratic)
//! - Sigma computation and scheduling
//! - Timestep shifting (SD3, dynamic resolution-dependent)
//! - Euler step execution
//! - CFG (Classifier-Free Guidance) application
//! - Add noise / scale noise operations
//! - Shift terminal stretching
//! - Edge cases and error handling
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_scheduler_parity.rs for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{DType, Device, Result, Tensor};
use candle_video::{RectifiedFlowScheduler, SchedulerConfig};

// ============================================================================
// Timestep Generation Tests
// ============================================================================

#[test]
fn test_linspace_timesteps_basic() {
    let config = SchedulerConfig {
        num_inference_steps: 50,
        guidance_scale: 3.0,
        timestep_spacing: "linspace".to_string(),
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let timesteps = scheduler.timesteps();

    assert_eq!(timesteps.len(), 50);
    // Timesteps should go from 1.0 (noise) to near 0.0 (clean)
    assert!(
        (timesteps[0] - 1.0).abs() < 1e-6,
        "First timestep should be 1.0, got {}",
        timesteps[0]
    );
    // Last timestep should be close to 0.0 but not exactly 0
    assert!(
        timesteps[49] < 0.1,
        "Last timestep should be close to 0, got {}",
        timesteps[49]
    );
}

#[test]
fn test_linspace_timesteps_small() {
    let config = SchedulerConfig {
        num_inference_steps: 5,
        guidance_scale: 3.0,
        timestep_spacing: "linspace".to_string(),
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let timesteps = scheduler.timesteps();

    assert_eq!(timesteps.len(), 5);
    // For 5 steps: [1.0, 0.8, 0.6, 0.4, 0.2] approximately
    assert!((timesteps[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_linspace_timesteps_single_step() {
    let config = SchedulerConfig {
        num_inference_steps: 1,
        guidance_scale: 3.0,
        timestep_spacing: "linspace".to_string(),
        use_dynamic_shifting: false, // Disable to prevent mu requirement
        shift_terminal: None,        // Disable stretching for this test
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let timesteps = scheduler.timesteps();

    assert_eq!(timesteps.len(), 1);
    assert!(
        (timesteps[0] - 1.0).abs() < 1e-6,
        "Single step should start at 1.0"
    );
}

#[test]
fn test_linear_quadratic_timesteps() {
    let config = SchedulerConfig {
        num_inference_steps: 20,
        guidance_scale: 3.0,
        timestep_spacing: "linear_quadratic".to_string(),
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let timesteps = scheduler.timesteps();

    assert_eq!(timesteps.len(), 20);
    // Linear quadratic schedule should be monotonically decreasing
    for i in 1..timesteps.len() {
        assert!(
            timesteps[i] < timesteps[i - 1],
            "Timesteps should be monotonically decreasing: t[{}]={} >= t[{}]={}",
            i,
            timesteps[i],
            i - 1,
            timesteps[i - 1]
        );
    }
}

#[test]
fn test_timesteps_are_decreasing() {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let timesteps = scheduler.timesteps();

    for i in 1..timesteps.len() {
        assert!(
            timesteps[i] < timesteps[i - 1],
            "Timesteps should be monotonically decreasing"
        );
    }
}

// ============================================================================
// Sigma Tests
// ============================================================================

#[test]
fn test_sigmas_computed_from_timesteps() {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let sigmas = scheduler.sigmas();
    let timesteps = scheduler.timesteps();

    assert_eq!(
        sigmas.len(),
        timesteps.len() + 1,
        "Sigmas should have terminal 0"
    );

    // Sigmas should match timesteps (in flow matching, sigma = t)
    for i in 0..timesteps.len() {
        assert!(
            (sigmas[i] - timesteps[i]).abs() < 1e-5,
            "Sigma[{}]={} should equal timestep[{}]={}",
            i,
            sigmas[i],
            i,
            timesteps[i]
        );
    }

    // Last sigma should be 0 (terminal)
    assert!(
        (sigmas[sigmas.len() - 1]).abs() < 1e-6,
        "Terminal sigma should be 0"
    );
}

#[test]
fn test_sigma_min_and_max() {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);

    let (sigma_min, sigma_max) = scheduler.sigma_range();
    assert!(
        sigma_min < sigma_max,
        "sigma_min should be less than sigma_max"
    );
    assert!(
        (sigma_max - 1.0).abs() < 0.1,
        "sigma_max should be close to 1.0"
    );
    assert!(sigma_min > 0.0, "sigma_min should be positive");
}

// ============================================================================
// Timestep Shifting Tests
// ============================================================================

#[test]
fn test_shift_parameter_applied() {
    let config_no_shift = SchedulerConfig {
        num_inference_steps: 10,
        shift: Some(1.0),            // No shift
        use_dynamic_shifting: false, // Static shift mode
        ..Default::default()
    };
    let config_with_shift = SchedulerConfig {
        num_inference_steps: 10,
        shift: Some(3.0),            // Apply shift
        use_dynamic_shifting: false, // Static shift mode
        ..Default::default()
    };

    let scheduler_no_shift = RectifiedFlowScheduler::new(config_no_shift);
    let scheduler_with_shift = RectifiedFlowScheduler::new(config_with_shift);

    let sigmas_no_shift = scheduler_no_shift.sigmas();
    let sigmas_with_shift = scheduler_with_shift.sigmas();

    // With shift > 1.0, sigmas should be shifted (higher values earlier)
    // The exact values depend on the shift formula: shift * s / (1 + (shift - 1) * s)
    assert!(
        (sigmas_no_shift[5] - sigmas_with_shift[5]).abs() > 1e-4,
        "Shift should affect sigma values"
    );
}

#[test]
fn test_dynamic_shifting_requires_mu() {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        use_dynamic_shifting: true,
        ..Default::default()
    };
    let mut scheduler = RectifiedFlowScheduler::new(config);

    // When using dynamic shifting, set_timesteps should require mu (resolution)
    let result = scheduler.set_timesteps_with_shape(10, None, None, None);
    assert!(result.is_err(), "Dynamic shifting requires mu parameter");
}

#[test]
fn test_resolution_dependent_shifting() {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        use_dynamic_shifting: true,
        ..Default::default()
    };
    let mut scheduler = RectifiedFlowScheduler::new(config);

    // Use shapes that produce token counts within the meaningful range
    // MIN_TOKENS = 1024, MAX_TOKENS = 4096
    // Small: 1 * 8 * 8 = 64 tokens (below MIN, clamped to base_shift)
    // Large: 8 * 16 * 16 = 2048 tokens (in range, interpolated)
    let shape_small = (1, 128, 1, 8, 8); // 1 * 8 * 8 = 64 tokens
    let shape_large = (1, 128, 8, 16, 16); // 8 * 16 * 16 = 2048 tokens

    scheduler
        .set_timesteps_with_shape(10, Some(shape_small), None, None)
        .unwrap();
    let sigmas_small = scheduler.sigmas().to_vec();

    scheduler
        .set_timesteps_with_shape(10, Some(shape_large), None, None)
        .unwrap();
    let sigmas_large = scheduler.sigmas().to_vec();

    // Larger resolution should shift timesteps differently
    assert!(
        (sigmas_small[5] - sigmas_large[5]).abs() > 1e-5,
        "Resolution should affect shifted sigmas: small[5]={}, large[5]={}",
        sigmas_small[5],
        sigmas_large[5]
    );
}

#[test]
fn test_shift_terminal_stretching() {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        shift_terminal: Some(0.1),
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let sigmas = scheduler.sigmas();

    // The terminal (last non-zero) sigma should be stretched to 0.1
    let last_sigma = sigmas[sigmas.len() - 2]; // Second to last (before terminal 0)
    assert!(
        (last_sigma - 0.1).abs() < 0.05,
        "Last sigma should be near shift_terminal 0.1, got {}",
        last_sigma
    );
}

// ============================================================================
// Euler Step Tests
// ============================================================================

#[test]
fn test_euler_step_basic() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        guidance_scale: 3.0,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;
    let model_output = Tensor::randn(0f32, 0.1, (1, 4, 8, 8), &device)?;

    let result = scheduler.step(&model_output, &sample, 0)?;
    assert_eq!(
        result.shape(),
        sample.shape(),
        "Output shape should match input"
    );

    Ok(())
}

#[test]
fn test_euler_step_formula() -> Result<()> {
    // Euler step: prev_sample = sample + dt * model_output
    // where dt = sigma_next - sigma (negative, as sigmas decrease)
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    // Use known values for verification
    let sample = Tensor::ones((1, 1, 1, 1), DType::F32, &device)?;
    let model_output = Tensor::ones((1, 1, 1, 1), DType::F32, &device)?;

    let sigmas = scheduler.sigmas();
    let dt = sigmas[1] - sigmas[0]; // Should be negative

    let result = scheduler.step(&model_output, &sample, 0)?;
    let expected_value = 1.0 + dt as f32;

    let result_val = result.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(
        (result_val - expected_value).abs() < 1e-5,
        "Expected {}, got {}",
        expected_value,
        result_val
    );

    Ok(())
}

#[test]
fn test_euler_step_all_timesteps() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let mut sample = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;

    // Step through all timesteps
    for i in 0..scheduler.num_steps() {
        let model_output = Tensor::randn(0f32, 0.1, (1, 4, 8, 8), &device)?;
        sample = scheduler.step(&model_output, &sample, i)?;
    }

    // Final sample should have same shape
    assert_eq!(sample.shape().dims(), &[1, 4, 8, 8]);
    Ok(())
}

#[test]
fn test_euler_step_out_of_bounds() {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device).unwrap();
    let model_output = Tensor::randn(0f32, 0.1, (1, 4, 8, 8), &device).unwrap();

    let result = scheduler.step(&model_output, &sample, 20);
    assert!(
        result.is_err(),
        "Should error on out-of-bounds timestep index"
    );
}

#[test]
fn test_step_with_timestep_tensor() -> Result<()> {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;
    let model_output = Tensor::randn(0f32, 0.1, (1, 4, 8, 8), &device)?;

    // Step with timestep value instead of index
    let timestep = scheduler.timesteps()[0];
    let result = scheduler.step_with_timestep(&model_output, &sample, timestep)?;

    assert_eq!(result.shape(), sample.shape());
    Ok(())
}

// ============================================================================
// Stochastic Sampling Tests
// ============================================================================

#[test]
fn test_stochastic_step_differs() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        stochastic_sampling: true,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let model_output = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;

    let result1 = scheduler.step(&model_output, &sample, 0)?;
    let result2 = scheduler.step(&model_output, &sample, 0)?;

    // Stochastic sampling should produce different results with noise
    let diff = result1
        .sub(&result2)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff > 0.0, "Stochastic sampling should add randomness");

    Ok(())
}

// ============================================================================
// CFG (Classifier-Free Guidance) Tests
// ============================================================================

#[test]
fn test_cfg_application() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 50,
        guidance_scale: 7.5,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let v_cond = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let v_uncond = Tensor::zeros((1, 4, 8, 8), DType::F32, &device)?;

    let v_guided = scheduler.apply_cfg(&v_cond, &v_uncond)?;

    // v_guided = v_uncond + scale * (v_cond - v_uncond)
    // v_guided = 0 + 7.5 * (1 - 0) = 7.5
    let result_val = v_guided.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(
        (result_val - 7.5).abs() < 1e-5,
        "CFG result should be 7.5, got {}",
        result_val
    );

    Ok(())
}

#[test]
fn test_cfg_with_negative_scale() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 50,
        guidance_scale: -1.0, // Negative guidance
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let v_cond = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let v_uncond = Tensor::zeros((1, 4, 8, 8), DType::F32, &device)?;

    let v_guided = scheduler.apply_cfg(&v_cond, &v_uncond)?;

    // v_guided = 0 + (-1) * (1 - 0) = -1
    let result_val = v_guided.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(
        (result_val - (-1.0)).abs() < 1e-5,
        "Negative CFG should invert direction"
    );

    Ok(())
}

#[test]
fn test_cfg_scale_one_is_conditional_only() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 50,
        guidance_scale: 1.0, // No guidance boost
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let v_cond = Tensor::full(5.0f32, (1, 4, 8, 8), &device)?;
    let v_uncond = Tensor::full(2.0f32, (1, 4, 8, 8), &device)?;

    let v_guided = scheduler.apply_cfg(&v_cond, &v_uncond)?;

    // v_guided = 2 + 1 * (5 - 2) = 5
    let result_val = v_guided.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(
        (result_val - 5.0).abs() < 1e-5,
        "Scale=1 should return conditional output"
    );

    Ok(())
}

// ============================================================================
// Add Noise / Scale Noise Tests
// ============================================================================

#[test]
fn test_add_noise_at_full_timestep() -> Result<()> {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let original = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let noise = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;

    // At t=1.0, noisy = (1-t)*original + t*noise = 0*original + 1*noise = noise
    let noisy = scheduler.add_noise(&original, &noise, 1.0)?;

    let diff = noisy.sub(&noise)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "At t=1.0, sample should be pure noise");

    Ok(())
}

#[test]
fn test_add_noise_at_zero_timestep() -> Result<()> {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let original = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let noise = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;

    // At t=0, noisy = (1-t)*original + t*noise = 1*original + 0*noise = original
    let noisy = scheduler.add_noise(&original, &noise, 0.0)?;

    let diff = noisy.sub(&original)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "At t=0, sample should be original");

    Ok(())
}

#[test]
fn test_add_noise_interpolation() -> Result<()> {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let original = Tensor::full(1.0f32, (1, 1), &device)?;
    let noise = Tensor::full(0.0f32, (1, 1), &device)?;

    // At t=0.5, noisy = 0.5 * original + 0.5 * noise = 0.5 * 1 + 0.5 * 0 = 0.5
    let noisy = scheduler.add_noise(&original, &noise, 0.5)?;

    let result_val = noisy.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(
        (result_val - 0.5).abs() < 1e-5,
        "At t=0.5, sample should be 0.5 interpolation"
    );

    Ok(())
}

#[test]
fn test_scale_model_input() -> Result<()> {
    // Flow matching schedulers typically don't scale input
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;
    let scaled = scheduler.scale_model_input(&sample, 0)?;

    // Should be identity operation
    let diff = scaled.sub(&sample)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-6, "scale_model_input should be identity");

    Ok(())
}

// ============================================================================
// Configuration and Accessor Tests
// ============================================================================

#[test]
fn test_scheduler_config_defaults() {
    let config = SchedulerConfig::default();
    assert_eq!(config.num_inference_steps, 50);
    assert!((config.guidance_scale - 3.0).abs() < 1e-6);
    assert_eq!(config.timestep_spacing, "linspace");
}

#[test]
fn test_num_steps_accessor() {
    let config = SchedulerConfig {
        num_inference_steps: 25,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    assert_eq!(scheduler.num_steps(), 25);
}

#[test]
fn test_guidance_scale_accessor() {
    let config = SchedulerConfig {
        guidance_scale: 5.5,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    assert!((scheduler.guidance_scale() - 5.5).abs() < 1e-6);
}

#[test]
fn test_init_noise_sigma() {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    assert!((scheduler.init_noise_sigma() - 1.0).abs() < 1e-6);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_zero_steps_returns_empty() {
    let config = SchedulerConfig {
        num_inference_steps: 0,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    assert!(scheduler.timesteps().is_empty());
}

#[test]
fn test_large_num_steps() {
    let config = SchedulerConfig {
        num_inference_steps: 1000,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let timesteps = scheduler.timesteps();

    assert_eq!(timesteps.len(), 1000);
    assert!((timesteps[0] - 1.0).abs() < 1e-5);
    // Check monotonicity
    for i in 1..timesteps.len() {
        assert!(timesteps[i] < timesteps[i - 1]);
    }
}

// ============================================================================
// Step Index Tracking Tests
// ============================================================================

#[test]
fn test_step_index_tracking() {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);

    assert!(
        scheduler.step_index().is_none(),
        "Step index should be None initially"
    );
}

#[test]
fn test_step_index_after_step() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let mut scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let model_output = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;

    scheduler.step_mut(&model_output, &sample, 0)?;
    assert_eq!(
        scheduler.step_index(),
        Some(1),
        "Step index should increment"
    );

    Ok(())
}

// ============================================================================
// Deterministic Seeding Tests
// ============================================================================

#[test]
#[ignore = "Seeded random generation not yet implemented"]
fn test_stochastic_with_seed() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        stochastic_sampling: true,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;
    let model_output = Tensor::ones((1, 4, 8, 8), DType::F32, &device)?;

    // With same seed, results should be reproducible
    let result1 = scheduler.step_with_seed(&model_output, &sample, 0, Some(42))?;
    let result2 = scheduler.step_with_seed(&model_output, &sample, 0, Some(42))?;

    let diff = result1
        .sub(&result2)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "Same seed should produce same result");

    Ok(())
}

// ============================================================================
// Integration with Pipeline Pattern Tests
// ============================================================================

#[test]
fn test_denoising_loop_pattern() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 5,
        guidance_scale: 3.0,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    // Initialize with noise
    let mut latents = Tensor::randn(0f32, 1.0, (1, 4, 16, 16), &device)?;

    // Simulate denoising loop
    for i in 0..scheduler.num_steps() {
        let _timestep = scheduler.timesteps()[i];

        // Simulate model output (conditional and unconditional)
        let noise_pred_cond = Tensor::randn(0f32, 0.1, latents.shape(), &device)?;
        let noise_pred_uncond = Tensor::randn(0f32, 0.1, latents.shape(), &device)?;

        // Apply CFG
        let noise_pred = scheduler.apply_cfg(&noise_pred_cond, &noise_pred_uncond)?;

        // Euler step
        latents = scheduler.step(&noise_pred, &latents, i)?;

        // Verify latents are valid
        assert!(latents.shape().dims() == &[1, 4, 16, 16]);
        // Check for finite values (no NaN or Inf)
        let sum = latents.sum_all()?.to_scalar::<f32>()?;
        assert!(
            sum.is_finite(),
            "Latents should not contain NaN after step {}",
            i
        );
    }

    Ok(())
}

// ============================================================================
// Output Structure Tests
// ============================================================================

#[test]
fn test_step_output() -> Result<()> {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    let sample = Tensor::randn(0f32, 1.0, (1, 4, 8, 8), &device)?;
    let model_output = Tensor::randn(0f32, 0.1, (1, 4, 8, 8), &device)?;

    let output = scheduler.step_output(&model_output, &sample, 0)?;

    assert!(output.prev_sample.shape().dims() == sample.shape().dims());
    // pred_original_sample may or may not be present

    Ok(())
}

// ============================================================================
// Per-Token Timestep Tests (Image-to-Video Conditioning)
// ============================================================================

#[test]
fn test_step_per_token_basic() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    // Create sample [B, seq, C]
    let sample = Tensor::randn(0f32, 1.0, (1, 16, 128), &device)?;
    let model_output = Tensor::randn(0f32, 0.1, (1, 16, 128), &device)?;

    // Per-token timesteps [B, seq] - all same value for uniform test
    let timestep = Tensor::full(0.8f32, (1, 16), &device)?;

    let result = scheduler.step_per_token(&model_output, &sample, &timestep)?;

    assert_eq!(
        result.dims(),
        sample.dims(),
        "Output shape should match input"
    );
    Ok(())
}

#[test]
fn test_step_per_token_different_timesteps() -> Result<()> {
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    // Sample [B=1, seq=4, C=8]
    let sample = Tensor::ones((1, 4, 8), DType::F32, &device)?;
    let model_output = Tensor::ones((1, 4, 8), DType::F32, &device)?;

    // Different timesteps per token: [1.0, 0.5, 0.1, 0.0]
    // Token at t=0.0 should not change (dt=0)
    let timesteps_data = vec![1.0f32, 0.5, 0.1, 0.0];
    let timestep = Tensor::from_vec(timesteps_data, (1, 4), &device)?;

    let result = scheduler.step_per_token(&model_output, &sample, &timestep)?;

    // The result should have different values per token due to different dt
    let result_data = result.flatten_all()?.to_vec1::<f32>()?;

    // Token at t=0.0 (last) should have dt=0, so result = sample = 1.0
    // Check last 8 values (last token's channels)
    let last_token_avg = result_data[24..32].iter().sum::<f32>() / 8.0;
    assert!(
        (last_token_avg - 1.0).abs() < 1e-4,
        "Token at t=0 should not change, got {}",
        last_token_avg
    );

    Ok(())
}

#[test]
fn test_add_noise_per_token() -> Result<()> {
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    // Original and noise [B, seq, C]
    let original = Tensor::ones((1, 4, 8), DType::F32, &device)?;
    let noise = Tensor::zeros((1, 4, 8), DType::F32, &device)?;

    // Per-token timesteps: [0.0, 0.5, 1.0, 0.25]
    let timesteps_data = vec![0.0f32, 0.5, 1.0, 0.25];
    let timesteps = Tensor::from_vec(timesteps_data, (1, 4), &device)?;

    let noisy = scheduler.add_noise_per_token(&original, &noise, &timesteps)?;
    let noisy_data = noisy.flatten_all()?.to_vec1::<f32>()?;

    // Token 0 at t=0.0: result = (1-0)*1 + 0*0 = 1.0
    let token0_avg = noisy_data[0..8].iter().sum::<f32>() / 8.0;
    assert!(
        (token0_avg - 1.0).abs() < 1e-5,
        "Token at t=0 should be original"
    );

    // Token 1 at t=0.5: result = 0.5*1 + 0.5*0 = 0.5
    let token1_avg = noisy_data[8..16].iter().sum::<f32>() / 8.0;
    assert!(
        (token1_avg - 0.5).abs() < 1e-5,
        "Token at t=0.5 should be 0.5"
    );

    // Token 2 at t=1.0: result = 0*1 + 1*0 = 0.0
    let token2_avg = noisy_data[16..24].iter().sum::<f32>() / 8.0;
    assert!(
        (token2_avg - 0.0).abs() < 1e-5,
        "Token at t=1.0 should be noise"
    );

    Ok(())
}

#[test]
fn test_per_token_conditioning_mask_pattern() -> Result<()> {
    // Simulate i2v conditioning: first tokens are conditioning (low t), rest are denoised (high t)
    let config = SchedulerConfig {
        num_inference_steps: 10,
        ..Default::default()
    };
    let scheduler = RectifiedFlowScheduler::new(config);
    let device = Device::Cpu;

    // 8 tokens: first 2 are conditioning (t=0 = clean), rest being denoised (t=0.9)
    let sample = Tensor::ones((1, 8, 16), DType::F32, &device)?;
    let model_output = Tensor::ones((1, 8, 16), DType::F32, &device)?;

    // Conditioning mask pattern: [0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    let timesteps_data = vec![0.0f32, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
    let timestep = Tensor::from_vec(timesteps_data, (1, 8), &device)?;

    let result = scheduler.step_per_token(&model_output, &sample, &timestep)?;
    let result_data = result.to_vec3::<f32>()?;

    // First 2 tokens (conditioning) should be unchanged
    for c in 0..16 {
        assert!(
            (result_data[0][0][c] - 1.0).abs() < 1e-4,
            "Conditioning token 0 should be unchanged"
        );
        assert!(
            (result_data[0][1][c] - 1.0).abs() < 1e-4,
            "Conditioning token 1 should be unchanged"
        );
    }

    // Remaining tokens should be modified
    let modified_token_val = result_data[0][2][0];
    assert!(
        (modified_token_val - 1.0).abs() > 0.01,
        "Denoising tokens should change, got {}",
        modified_token_val
    );

    Ok(())
}
