//! Tests for VAE Encoder
//!
//! Tests for the VAE encoder components used in image-to-video conditioning.
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_vae_parity.rs for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_video::config::VaeConfig;
use candle_video::vae::{LatentLogVar, patchify, unpatchify};

// =============================================================================
// Helper functions
// =============================================================================

fn create_test_device() -> Device {
    Device::Cpu
}

// =============================================================================
// Patchify/Unpatchify Tests
// =============================================================================

#[test]
fn test_patchify_basic() -> Result<()> {
    let device = create_test_device();

    // Input: [B=1, C=3, T=2, H=8, W=8]
    let input = Tensor::randn(0f32, 1.0, (1, 3, 2, 8, 8), &device)?;

    // Patchify with patch_size_hw=2, patch_size_t=1
    let patched = patchify(&input, 2, 1)?;

    // Expected: [B=1, C=3*4=12, T=2, H=4, W=4]
    assert_eq!(patched.dims(), &[1, 12, 2, 4, 4]);

    Ok(())
}

#[test]
fn test_patchify_with_temporal() -> Result<()> {
    let device = create_test_device();

    // Input: [B=1, C=3, T=4, H=8, W=8]
    let input = Tensor::randn(0f32, 1.0, (1, 3, 4, 8, 8), &device)?;

    // Patchify with patch_size_hw=2, patch_size_t=2
    let patched = patchify(&input, 2, 2)?;

    // Expected: [B=1, C=3*2*4=24, T=2, H=4, W=4]
    assert_eq!(patched.dims(), &[1, 24, 2, 4, 4]);

    Ok(())
}

#[test]
fn test_patchify_identity() -> Result<()> {
    let device = create_test_device();

    // Input: [B=1, C=4, T=2, H=8, W=8]
    let input = Tensor::randn(0f32, 1.0, (1, 4, 2, 8, 8), &device)?;

    // Patchify with patch_size=1 (no-op)
    let patched = patchify(&input, 1, 1)?;

    // Should be same shape
    assert_eq!(patched.dims(), input.dims());

    Ok(())
}

#[test]
fn test_unpatchify_basic() -> Result<()> {
    let device = create_test_device();

    // Input: [B=1, C=12, T=2, H=4, W=4] (patched)
    let patched = Tensor::randn(0f32, 1.0, (1, 12, 2, 4, 4), &device)?;

    // Unpatchify with patch_size_hw=2, patch_size_t=1
    let output = unpatchify(&patched, 2, 1)?;

    // Expected: [B=1, C=3, T=2, H=8, W=8]
    assert_eq!(output.dims(), &[1, 3, 2, 8, 8]);

    Ok(())
}

#[test]
fn test_patchify_unpatchify_roundtrip() -> Result<()> {
    let device = create_test_device();

    // Input: [B=1, C=3, T=2, H=8, W=8]
    let input = Tensor::randn(0f32, 1.0, (1, 3, 2, 8, 8), &device)?;

    // Patchify then unpatchify
    let patched = patchify(&input, 2, 1)?;
    let output = unpatchify(&patched, 2, 1)?;

    // Should recover original shape
    assert_eq!(output.dims(), input.dims());

    // Check values are preserved (within tolerance)
    let max_diff = (input.sub(&output)?)
        .abs()?
        .max(0)?
        .max(0)?
        .max(0)?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(max_diff < 1e-5, "Max diff: {}", max_diff);

    Ok(())
}

#[test]
fn test_unpatchify_upsampler_axis_order_matches_diffusers() -> Result<()> {
    let device = create_test_device();
    // This test locks in the spatial patch axis order for the VAE upsampler.
    //
    // The LTX upsampler performs a 3D "depth-to-space" style unpacking with stride (2,2,2).
    // In diffusers, spatial unpacking pairs:
    // - height with `stride_h`
    // - width  with `stride_w`
    //
    // Swapping stride_h/stride_w produces a characteristic checkerboard / grid artifact.

    let (s_t, s_h, s_w) = (2usize, 2usize, 2usize);
    let (b, c, t, h, w) = (1usize, 1usize, 2usize, 1usize, 1usize);
    let c_packed = c * s_t * s_h * s_w;

    // Fill input so each packed channel is uniquely identifiable per source timestep.
    // value = src_t * 100 + packed_channel
    let mut data = Vec::with_capacity(b * c_packed * t * h * w);
    for _bi in 0..b {
        for packed_ch in 0..c_packed {
            for src_t in 0..t {
                for _hi in 0..h {
                    for _wi in 0..w {
                        data.push((src_t * 100 + packed_ch) as f32);
                    }
                }
            }
        }
    }
    let input = Tensor::from_vec(data, (b, c_packed, t, h, w), &device)?;

    let output = unpatchify(&input, s_h, s_t)?;

    let out_t = t * s_t - (s_t - 1);
    assert_eq!(output.dims(), &[b, c, out_t, h * s_h, w * s_w]);

    for t_out in 0..out_t {
        for y_out in 0..(h * s_h) {
            for x_out in 0..(w * s_w) {
                // Undo the causal slice: output time index corresponds to t_total = t_out + (s_t - 1).
                let t_total = t_out + (s_t - 1);
                let src_t = t_total / s_t;
                let t_off = t_total % s_t;

                let h_off = y_out % s_h;
                let w_off = x_out % s_w;

                // Packing order: (c, t_off, h_off, w_off) with w_off as the innermost component.
                let packed_ch = (((0 * s_t + t_off) * s_h + h_off) * s_w + w_off) as f32;
                let expected = src_t as f32 * 100.0 + packed_ch;

                let actual = output.i((0, 0, t_out, y_out, x_out))?.to_scalar::<f32>()?;
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "mismatch at t_out={t_out}, y_out={y_out}, x_out={x_out}: expected {expected}, got {actual}"
                );
            }
        }
    }

    Ok(())
}

// =============================================================================
// LatentLogVar Tests
// =============================================================================

#[test]
fn test_latent_log_var_enum() {
    assert_eq!(LatentLogVar::default(), LatentLogVar::None);

    let per_channel = LatentLogVar::PerChannel;
    let uniform = LatentLogVar::Uniform;
    let constant = LatentLogVar::Constant;

    assert_ne!(per_channel, uniform);
    assert_ne!(uniform, constant);
}

// =============================================================================
// VaeConfig Tests
// =============================================================================

#[test]
fn test_vae_config_default() {
    let config = VaeConfig::default();

    assert_eq!(config.in_channels, 3);
    assert_eq!(config.out_channels, 3);
    assert_eq!(config.latent_channels, 128);
}

#[test]
fn test_vae_config_custom() {
    let config = VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 64,
        block_out_channels: vec![128, 256, 512],
        layers_per_block: 2,
        temporal_downsample: 8,
        spatial_downsample: 32,
        causal: true,
        latents_mean: None,
        latents_std: None,
        scaling_factor: 1.0,
        timestep_conditioning: false,
    };

    assert_eq!(config.latent_channels, 64);
    assert_eq!(config.temporal_downsample, 8);
}
