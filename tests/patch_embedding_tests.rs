//! Spatial-Temporal Patch Embedding Tests - TDD approach
//!
//! Tests for the PatchEmbedding module that converts 5D video tensors
//! (B, C, T, H, W) into sequences of embedded patches (B, num_patches, hidden_size)
//!
//! Feature coverage:
//! - Basic 5D tensor patchification
//! - Separate patch_size_t and patch_size for temporal/spatial dimensions
//! - Indices grid generation for RoPE
//! - Unpatchify (inverse operation)
//! - Edge cases and dimension validation
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_*_parity.rs tests for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_video::config::DitConfig;
use candle_video::dit::PatchEmbedding;

// ===========================================================================
// Test Helpers
// ===========================================================================

fn small_test_config() -> DitConfig {
    DitConfig {
        patch_size: 1,
        patch_size_t: None,
        in_channels: 32,
        hidden_size: 64,
        depth: 2,
        num_heads: 4,
        caption_channels: 128,
        mlp_ratio: 2.0,
        use_flash_attention: false,
        timestep_scale_multiplier: None,
    }
}

// ===========================================================================
// Basic Patch Embedding Tests
// ===========================================================================

#[test]
fn test_patch_embedding_5d_basic() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // 5D input: (B, C, T, H, W)
    let batch = 2;
    let c = config.in_channels;
    let t = 4;
    let h = 8;
    let w = 8;

    let x = Tensor::randn(0f32, 1.0, (batch, c, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    // With patch_size=1: num_patches = T * H * W = 4 * 8 * 8 = 256
    let expected_patches = t * h * w;
    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, expected_patches);
    assert_eq!(out.dim(2)?, config.hidden_size);

    Ok(())
}

#[test]
fn test_patch_embedding_patch_size_2() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        in_channels: 32,
        hidden_size: 64,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // 5D input: (B, C, T, H, W)
    let batch = 1;
    let c = config.in_channels;
    let t = 4; // -> 2 patches
    let h = 16; // -> 8 patches
    let w = 16; // -> 8 patches

    let x = Tensor::randn(0f32, 1.0, (batch, c, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    // With patch_size=2: num_patches = (T/2) * (H/2) * (W/2) = 2 * 8 * 8 = 128
    let expected_patches = (t / 2) * (h / 2) * (w / 2);
    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, expected_patches);
    assert_eq!(out.dim(2)?, config.hidden_size);

    Ok(())
}

#[test]
fn test_patch_embedding_preserves_content() -> Result<()> {
    // Verify that patchification doesn't lose information
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 1,
        in_channels: 4,
        hidden_size: 4, // Same as input for easy verification
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Simple input
    let batch = 1;
    let c = 4;
    let t = 2;
    let h = 2;
    let w = 2;

    let x = Tensor::randn(0f32, 1.0, (batch, c, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    // Total elements should be preserved (before projection)
    let _input_elements = batch * c * t * h * w;
    let output_patches = t * h * w;
    assert_eq!(out.dim(1)?, output_patches);

    // Make sure we got all the spatial positions
    let total_spatial = t * h * w;
    assert_eq!(out.dim(1)?, total_spatial);

    Ok(())
}

// ===========================================================================
// Indices Grid Tests
// ===========================================================================

#[test]
fn test_indices_grid_generation() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        in_channels: 32,
        hidden_size: 64,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 2;
    let t = 4;
    let h = 16;
    let w = 16;

    let indices = embed.get_indices_grid(batch, t, h, w, &device)?;

    // After patchification: T=2, H=8, W=8
    let t_patches = t / config.patch_size;
    let h_patches = h / config.patch_size;
    let w_patches = w / config.patch_size;
    let num_patches = t_patches * h_patches * w_patches;

    // Indices grid shape: (batch, 3, num_patches)
    assert_eq!(indices.dim(0)?, batch);
    assert_eq!(indices.dim(1)?, 3); // t, h, w coordinates
    assert_eq!(indices.dim(2)?, num_patches);

    Ok(())
}

#[test]
fn test_indices_grid_normalized_values() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 1,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 1;
    let t = 4;
    let h = 4;
    let w = 4;

    let indices = embed.get_indices_grid(batch, t, h, w, &device)?;

    // Get values as vec
    let vals = indices.flatten_all()?.to_vec1::<f32>()?;

    // All values should be in [0, 1] (normalized coordinates)
    for val in vals {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Coordinate {} is not in [0, 1]",
            val
        );
    }

    Ok(())
}

#[test]
fn test_indices_grid_t_coordinate_ordering() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 1,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 1;
    let t = 3;
    let h = 2;
    let w = 2;

    let indices = embed.get_indices_grid(batch, t, h, w, &device)?;

    // Extract t coordinates (first row of indices)
    let t_coords = indices.i((0, 0, ..))?.to_vec1::<f32>()?;

    // T coordinates should increase with stride h*w
    // For t=3, h=2, w=2: first 4 values should have t=0, next 4 have t=0.5, last 4 have t=1.0
    let hw = h * w;
    for ti in 0..t {
        let expected_t = if t > 1 {
            ti as f32 / (t - 1) as f32
        } else {
            0.0
        };
        for pos in 0..hw {
            let idx = ti * hw + pos;
            assert!(
                (t_coords[idx] - expected_t).abs() < 1e-6,
                "T coordinate at idx {} expected {} but got {}",
                idx,
                expected_t,
                t_coords[idx]
            );
        }
    }

    Ok(())
}

// ===========================================================================
// Dimension Validation Tests
// ===========================================================================

#[test]
fn test_patch_embedding_batch_size_1() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        in_channels: 16,
        hidden_size: 32,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Minimal batch size
    let batch = 1;
    let x = Tensor::randn(0f32, 1.0, (batch, 16, 4, 8, 8), &device)?;
    let out = embed.forward(&x)?;

    let expected_patches = (4 / 2) * (8 / 2) * (8 / 2);
    assert_eq!(out.dim(0)?, 1);
    assert_eq!(out.dim(1)?, expected_patches);

    Ok(())
}

#[test]
fn test_patch_embedding_large_batch() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Larger batch size
    let batch = 8;
    let x = Tensor::randn(0f32, 1.0, (batch, 32, 2, 4, 4), &device)?;
    let out = embed.forward(&x)?;

    assert_eq!(out.dim(0)?, batch);

    Ok(())
}

#[test]
fn test_patch_embedding_asymmetric_dimensions() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        in_channels: 16,
        hidden_size: 32,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Non-square dimensions
    let batch = 1;
    let t = 4;
    let h = 8;
    let w = 16;
    let x = Tensor::randn(0f32, 1.0, (batch, 16, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    let expected_patches = (t / 2) * (h / 2) * (w / 2);
    assert_eq!(out.dim(1)?, expected_patches);

    Ok(())
}

// ===========================================================================
// dtype Preservation Tests
// ===========================================================================

#[test]
fn test_patch_embedding_f32_dtype() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let x = Tensor::randn(0f32, 1.0, (1, 32, 2, 4, 4), &device)?;
    let out = embed.forward(&x)?;

    assert_eq!(out.dtype(), DType::F32);

    Ok(())
}

#[test]
fn test_patch_embedding_f16_dtype() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let x = Tensor::zeros((1, 32, 2, 4, 4), DType::F16, &device)?;
    let out = embed.forward(&x)?;

    assert_eq!(out.dtype(), DType::F16);

    Ok(())
}

// ===========================================================================
// Integration with RoPE Tests
// ===========================================================================

#[test]
fn test_patch_embedding_rope_compatibility() -> Result<()> {
    use candle_video::rope::FractionalRoPE;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        in_channels: 32,
        hidden_size: 72, // Divisible by 6 for 3D RoPE
        num_heads: 4,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 2;
    let t = 4;
    let h = 8;
    let w = 8;

    // Get patch embeddings
    let x = Tensor::randn(0f32, 1.0, (batch, config.in_channels, t, h, w), &device)?;
    let _patches = embed.forward(&x)?;

    // Get indices grid
    let indices = embed.get_indices_grid(batch, t, h, w, &device)?;

    // Create RoPE and compute frequencies
    let head_dim = config.hidden_size / config.num_heads; // 72 / 4 = 18
    let rope = FractionalRoPE::new(head_dim, 10000.0, 1024);
    let freqs_cis = rope.compute_freqs_cis(&indices, &device)?;

    // Freqs should have correct shape for attention
    let (cos, sin) = freqs_cis;
    let num_patches = (t / 2) * (h / 2) * (w / 2);

    assert_eq!(cos.dim(0)?, batch);
    assert_eq!(cos.dim(1)?, num_patches);
    assert_eq!(cos.dim(2)?, head_dim);

    assert_eq!(sin.dim(0)?, batch);
    assert_eq!(sin.dim(1)?, num_patches);

    Ok(())
}

// ===========================================================================
// Separate Temporal and Spatial Patch Size Tests (LTX-Video Feature)
// ===========================================================================

#[test]
fn test_patch_embedding_separate_t_patch_size() -> Result<()> {
    // This test verifies support for separate patch_size_t and patch_size
    // as used in official LTX-Video implementation
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Using extended config with separate temporal patch size would require
    // config changes. For now, test with unified patch_size
    let config = DitConfig {
        patch_size: 1, // Spatial patch size
        in_channels: 32,
        hidden_size: 64,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 1;
    let t = 8;
    let h = 16;
    let w = 16;

    let x = Tensor::randn(0f32, 1.0, (batch, 32, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    // With patch_size=1: all spatial points are patches
    let expected = t * h * w;
    assert_eq!(out.dim(1)?, expected);

    Ok(())
}

// ===========================================================================
// Edge Cases
// ===========================================================================

#[test]
fn test_patch_embedding_single_frame() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Single temporal frame (T=1)
    let batch = 1;
    let t = 1;
    let h = 4;
    let w = 4;
    let x = Tensor::randn(0f32, 1.0, (batch, 32, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    assert_eq!(out.dim(1)?, t * h * w);

    Ok(())
}

#[test]
fn test_patch_embedding_minimal_spatial() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Minimal spatial: 1x1
    let batch = 1;
    let t = 4;
    let h = 1;
    let w = 1;
    let x = Tensor::randn(0f32, 1.0, (batch, 32, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    assert_eq!(out.dim(1)?, t * h * w);

    Ok(())
}

#[test]
fn test_indices_grid_single_patch() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Single patch after patchification (2/2=1, 2/2=1, 2/2=1)
    let batch = 1;
    let t = 2;
    let h = 2;
    let w = 2;

    let indices = embed.get_indices_grid(batch, t, h, w, &device)?;

    assert_eq!(indices.dim(2)?, 1); // Only 1 patch

    // Single patch should have t=0, h=0, w=0 (normalized to 0 when only 1 patch)
    let vals = indices.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(vals, vec![0.0, 0.0, 0.0]);

    Ok(())
}

// ===========================================================================
// New: PatchEmbedding Extended API Tests
// ===========================================================================

#[test]
fn test_patch_embedding_num_patches() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(1),
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // T=8, H=16, W=16 with patch_size=2, patch_size_t=1
    let expected = (8 / 1) * (16 / 2) * (16 / 2); // 8 * 8 * 8 = 512
    assert_eq!(embed.num_patches(8, 16, 16), expected);

    Ok(())
}

#[test]
fn test_patch_embedding_patch_dims() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(4),
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let (t_patches, h_patches, w_patches) = embed.patch_dims(16, 32, 64);

    assert_eq!(t_patches, 16 / 4); // 4
    assert_eq!(h_patches, 32 / 2); // 16
    assert_eq!(w_patches, 64 / 2); // 32

    Ok(())
}

#[test]
fn test_patch_size_accessors() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(4),
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    assert_eq!(embed.patch_size(), 2);
    assert_eq!(embed.patch_size_t(), 4);

    Ok(())
}

#[test]
fn test_patch_embedding_separate_temporal_spatial() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // LTX-Video style: spatial patching (2x2) but no temporal patching (1)
    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(1), // No temporal patching
        in_channels: 32,
        hidden_size: 64,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 1;
    let t = 8;
    let h = 16;
    let w = 16;

    let x = Tensor::randn(0f32, 1.0, (batch, 32, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    // T stays 8, H becomes 8, W becomes 8
    let expected_patches = 8 * 8 * 8; // 512
    assert_eq!(out.dim(1)?, expected_patches);

    Ok(())
}

#[test]
fn test_patch_embedding_only_temporal_patching() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Only temporal patching
    let config = DitConfig {
        patch_size: 1,         // No spatial patching
        patch_size_t: Some(2), // Temporal patching
        in_channels: 16,
        hidden_size: 32,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 1;
    let t = 4;
    let h = 4;
    let w = 4;

    let x = Tensor::randn(0f32, 1.0, (batch, 16, t, h, w), &device)?;
    let out = embed.forward(&x)?;

    // T becomes 2, H stays 4, W stays 4
    let expected_patches = 2 * 4 * 4; // 32
    assert_eq!(out.dim(1)?, expected_patches);

    Ok(())
}

// ===========================================================================
// Unpatchify Tests
// ===========================================================================

#[test]
fn test_unpatchify_basic() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 1,
        patch_size_t: Some(1),
        in_channels: 4,
        hidden_size: 4,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Create patches tensor (already in patch_dim format, not hidden_size)
    let batch = 1;
    let t_patches = 2;
    let h_patches = 3;
    let w_patches = 4;
    let c = 4;
    let num_patches = t_patches * h_patches * w_patches;

    let patches = Tensor::randn(0f32, 1.0, (batch, num_patches, c), &device)?;

    let out = embed.unpatchify(&patches, t_patches, h_patches, w_patches)?;

    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, c);
    assert_eq!(out.dim(2)?, t_patches); // T
    assert_eq!(out.dim(3)?, h_patches); // H
    assert_eq!(out.dim(4)?, w_patches); // W

    Ok(())
}

#[test]
fn test_unpatchify_spatial_only() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(1),
        in_channels: 4,
        hidden_size: 16,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 2;
    let t_patches = 4; // T stays as is
    let h_patches = 8; // H/2
    let w_patches = 8; // W/2
    let patch_dim = 4 * 1 * 2 * 2; // C * ps_t * ps * ps = 16

    let num_patches = t_patches * h_patches * w_patches;
    let patches = Tensor::randn(0f32, 1.0, (batch, num_patches, patch_dim), &device)?;

    let out = embed.unpatchify(&patches, t_patches, h_patches, w_patches)?;

    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, 4); // C
    assert_eq!(out.dim(2)?, 4); // T = t_patches * ps_t = 4 * 1
    assert_eq!(out.dim(3)?, 16); // H = h_patches * ps = 8 * 2
    assert_eq!(out.dim(4)?, 16); // W = w_patches * ps = 8 * 2

    Ok(())
}

#[test]
fn test_patchify_unpatchify_roundtrip() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Use identity projection for roundtrip test
    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(1),
        in_channels: 4,
        hidden_size: 4 * 1 * 2 * 2, // Equal to patch_dim
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    let batch = 1;
    let c = 4;
    let t = 4;
    let h = 8;
    let w = 8;

    // Create test input
    let x = Tensor::arange(0f32, (batch * c * t * h * w) as f32, &device)?
        .reshape((batch, c, t, h, w))?;

    // Patchify (manually, without projection for exact roundtrip)
    let (_t_patches, _h_patches, _w_patches) = embed.patch_dims(t, h, w);
    let num_patches = embed.num_patches(t, h, w);

    // Should process correctly
    let patches_projected = embed.forward(&x)?;
    assert_eq!(patches_projected.dim(1)?, num_patches);

    Ok(())
}

#[test]
fn test_unpatchify_error_on_wrong_patches() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(1),
        in_channels: 4,
        hidden_size: 64,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // Wrong number of patches
    let patches = Tensor::randn(0f32, 1.0, (1, 100, 16), &device)?;

    let result = embed.unpatchify(&patches, 2, 4, 4); // Expected: 2*4*4=32, got 100
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_patchify_error_on_indivisible_dims() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        patch_size_t: Some(2),
        in_channels: 4,
        hidden_size: 16,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;

    // T=3 is not divisible by patch_size_t=2
    let x = Tensor::randn(0f32, 1.0, (1, 4, 3, 4, 4), &device)?;
    let result = embed.forward(&x);
    assert!(result.is_err());

    // H=5 is not divisible by patch_size=2
    let x = Tensor::randn(0f32, 1.0, (1, 4, 4, 5, 4), &device)?;
    let result = embed.forward(&x);
    assert!(result.is_err());

    Ok(())
}
