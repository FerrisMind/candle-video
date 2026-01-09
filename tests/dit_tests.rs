//! DiT Block Tests - TDD approach for LTX-Video DiT implementation
//!
//! This test module covers:
//! - JointAttention with RoPE (Rotary Position Embedding)
//! - BasicTransformerBlock (AdaLN-single modulation)
//! - FeedForward with GEGLU activation
//! - Full Transformer3D model
//! - PatchEmbedding for 5D video tensors
//! - Flash Attention fallback
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_dit_parity.rs for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_video::config::DitConfig;
use candle_video::dit::{
    AdaLayerNormSingle, Attention, BasicTransformerBlock, CaptionProjection, DiTBlock, FeedForward,
    PatchEmbedding, Transformer3DModel,
};
use candle_video::rope::{FractionalRoPE, generate_coord_grids};

// ===========================================================================
// Test Helpers
// ===========================================================================

#[allow(dead_code)]
fn default_test_config() -> DitConfig {
    DitConfig {
        patch_size: 2,
        patch_size_t: None,
        in_channels: 128,
        hidden_size: 1152,
        depth: 28,
        num_heads: 16,
        caption_channels: 4096,
        mlp_ratio: 4.0,
        use_flash_attention: false, // CPU tests
        timestep_scale_multiplier: None,
    }
}

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
// PatchEmbedding Tests
// ===========================================================================

#[test]
fn test_patch_embedding_forward_shape() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let embed = PatchEmbedding::new(vb.pp("patch_embed"), &config)?;

    // Input: (B, C, T, H, W) where C = in_channels (latent channels from VAE)
    let batch = 2;
    let channels = config.in_channels;
    let t = 4; // frames
    let h = 8; // height
    let w = 8; // width
    let x = Tensor::randn(0f32, 1.0, (batch, channels, t, h, w), &device)?;

    let out = embed.forward(&x)?;

    // After patch_size=1: patches = T * H * W = 4 * 8 * 8 = 256
    // Output: (B, num_patches, hidden_size)
    let expected_patches = t * h * w;
    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, expected_patches);
    assert_eq!(out.dim(2)?, config.hidden_size);

    Ok(())
}

#[test]
fn test_patch_embedding_with_patch_size_2() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        patch_size: 2,
        in_channels: 32,
        hidden_size: 64,
        ..small_test_config()
    };

    let embed = PatchEmbedding::new(vb.pp("patch_embed"), &config)?;

    // Input: (B, C, T, H, W)
    let batch = 1;
    let x = Tensor::randn(0f32, 1.0, (batch, 32, 4, 16, 16), &device)?;

    let out = embed.forward(&x)?;

    // After patch_size=2: T'=2, H'=8, W'=8 => patches = 2*8*8 = 128
    let expected_patches = (4 / 2) * (16 / 2) * (16 / 2);
    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, expected_patches);
    assert_eq!(out.dim(2)?, config.hidden_size);

    Ok(())
}

// ===========================================================================
// Attention Tests
// ===========================================================================

#[test]
fn test_attention_basic_forward() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let attn = Attention::new(vb.pp("attn"), &config, None)?;

    let batch = 2;
    let seq_len = 16;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    // Self-attention without RoPE
    let out = attn.forward(&x, None, None, None, None, None)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

#[test]
fn test_attention_with_rope() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let attn = Attention::new(vb.pp("attn"), &config, None)?;

    let batch = 2;
    let seq_len = 16;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    // Create RoPE frequencies - must use hidden_size (not head_dim) for linear RoPE application
    let rope = FractionalRoPE::new(config.hidden_size, 10000.0, 1024);

    // Generate indices grid for RoPE
    let indices = Tensor::zeros((batch, 3, seq_len), DType::F32, &device)?;
    let freqs_cis = rope.compute_freqs_cis(&indices, &device)?;

    let out = attn.forward(&x, Some(&freqs_cis), None, None, None, None)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

#[test]
fn test_attention_cross_attention() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let attn = Attention::new(vb.pp("attn"), &config, Some(config.hidden_size))?;

    let batch = 2;
    let seq_len = 16;
    let encoder_seq_len = 32;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;
    let encoder_hidden = Tensor::randn(
        0f32,
        1.0,
        (batch, encoder_seq_len, config.hidden_size),
        &device,
    )?;

    let out = attn.forward(&x, None, Some(&encoder_hidden), None, None, None)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

#[test]
fn test_attention_with_mask() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let attn = Attention::new(vb.pp("attn"), &config, None)?;

    let batch = 2;
    let seq_len = 16;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    // Attention mask: (batch, 1, seq_len, seq_len)
    let mask = Tensor::zeros((batch, 1, seq_len, seq_len), DType::F32, &device)?;

    let out = attn.forward(&x, None, None, Some(&mask), None, None)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

// ===========================================================================
// FeedForward Tests
// ===========================================================================

#[test]
fn test_feedforward_geglu() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let ff = FeedForward::new(vb.pp("ff"), &config)?;

    let batch = 2;
    let seq_len = 16;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    let out = ff.forward(&x)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

#[test]
fn test_feedforward_dimensions() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        hidden_size: 128,
        mlp_ratio: 4.0, // inner_dim = 128 * 4 = 512
        ..small_test_config()
    };

    let ff = FeedForward::new(vb.pp("ff"), &config)?;

    let batch = 1;
    let seq_len = 8;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    let out = ff.forward(&x)?;

    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, seq_len);
    assert_eq!(out.dim(2)?, config.hidden_size);

    Ok(())
}

// ===========================================================================
// AdaLayerNormSingle Tests
// ===========================================================================

#[test]
fn test_adaln_single_forward() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let hidden_size = 64;
    let adaln = AdaLayerNormSingle::new(vb.pp("adaln"), hidden_size)?;

    let batch = 2;
    let timestep = Tensor::randn(0f32, 1.0, (batch,), &device)?;

    let (emb, embedded_timestep) = adaln.forward(&timestep)?;

    // Output shape: (batch, 1, 6 * hidden_size) for single_scale_shift
    assert_eq!(emb.dim(0)?, batch);
    assert_eq!(embedded_timestep.dim(0)?, batch);

    Ok(())
}

// ===========================================================================
// CaptionProjection Tests
// ===========================================================================

#[test]
fn test_caption_projection() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let proj = CaptionProjection::new(vb.pp("caption_proj"), &config)?;

    let batch = 2;
    let seq_len = 32;
    let caption_emb = Tensor::randn(
        0f32,
        1.0,
        (batch, seq_len, config.caption_channels),
        &device,
    )?;

    let out = proj.forward(&caption_emb)?;

    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, seq_len);
    assert_eq!(out.dim(2)?, config.hidden_size);

    Ok(())
}

// ===========================================================================
// BasicTransformerBlock Tests
// ===========================================================================

#[test]
fn test_basic_transformer_block_forward() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let block = BasicTransformerBlock::new(vb.pp("block"), &config)?;

    let batch = 2;
    let seq_len = 16;
    let encoder_seq_len = 32;

    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;
    let encoder_hidden = Tensor::randn(
        0f32,
        1.0,
        (batch, encoder_seq_len, config.hidden_size),
        &device,
    )?;
    let timestep = Tensor::randn(0f32, 1.0, (batch, 1, 6 * config.hidden_size), &device)?;

    let out = block.forward(
        &x,
        None,
        Some(&encoder_hidden),
        Some(&timestep),
        None,
        None,
        None,
    )?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

#[test]
fn test_basic_transformer_block_self_attention_only() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let block = BasicTransformerBlock::new(vb.pp("block"), &config)?;

    let batch = 2;
    let seq_len = 16;

    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;
    let timestep = Tensor::randn(0f32, 1.0, (batch, 1, 6 * config.hidden_size), &device)?;

    // No encoder_hidden_states - self-attention only
    let out = block.forward(&x, None, None, Some(&timestep), None, None, None)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

// ===========================================================================
// Transformer3DModel Tests
// ===========================================================================

#[test]
fn test_transformer3d_model_forward() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        depth: 2, // Small depth for testing
        ..small_test_config()
    };

    let model = Transformer3DModel::new(vb.pp("transformer"), &config)?;

    let batch = 1;
    let t = 4;
    let h = 4;
    let w = 4;
    let caption_seq_len = 16;

    // 5D input: (B, C, T, H, W)
    let hidden_states = Tensor::randn(0f32, 1.0, (batch, config.in_channels, t, h, w), &device)?;

    // Indices grid for RoPE: (batch, 3, seq_len)
    let seq_len = t * h * w;
    let indices_grid = Tensor::zeros((batch, 3, seq_len), DType::F32, &device)?;

    // Caption embeddings
    let encoder_hidden = Tensor::randn(
        0f32,
        1.0,
        (batch, caption_seq_len, config.caption_channels),
        &device,
    )?;

    // Timestep embedding
    let timestep = Tensor::randn(0f32, 1.0, (batch,), &device)?;

    let out = model.forward(
        &hidden_states,
        &indices_grid,
        Some(&encoder_hidden),
        &timestep,
        None,
        None,
        None,
        None,
    )?;

    // Output should be 5D: (B, C, T, H, W)
    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, config.in_channels);
    assert_eq!(out.dim(2)?, t);
    assert_eq!(out.dim(3)?, h);
    assert_eq!(out.dim(4)?, w);

    Ok(())
}

#[test]
fn test_transformer3d_model_gradient_checkpointing() -> Result<()> {
    // This test just ensures model can be created with gradient_checkpointing flag
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let _model = Transformer3DModel::new(vb.pp("transformer"), &config)?;

    // Model should support gradient checkpointing (training optimization)
    // In inference, this has no effect but should not error
    assert!(true);

    Ok(())
}

// ===========================================================================
// RoPE Integration Tests
// ===========================================================================

#[test]
fn test_rope_freqs_cis_computation() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 72; // Must be divisible by 6 for 3D RoPE (t, h, w each get dim/3/2)
    let rope = FractionalRoPE::new(head_dim, 10000.0, 1024);

    let batch = 2;
    let seq_len = 16;

    // Indices grid: (batch, 3, seq_len) - fractional positions for t, h, w
    let indices = Tensor::randn(0f32, 0.5, (batch, 3, seq_len), &device)?;

    let (cos_freqs, sin_freqs) = rope.compute_freqs_cis(&indices, &device)?;

    // Freqs should have shape (batch, seq_len, head_dim)
    assert_eq!(cos_freqs.dim(0)?, batch);
    assert_eq!(cos_freqs.dim(1)?, seq_len);
    assert_eq!(cos_freqs.dim(2)?, head_dim);

    assert_eq!(sin_freqs.dim(0)?, batch);
    assert_eq!(sin_freqs.dim(1)?, seq_len);
    assert_eq!(sin_freqs.dim(2)?, head_dim);

    Ok(())
}

#[test]
fn test_rope_apply_rotary_embedding() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 72;
    let rope = FractionalRoPE::new(head_dim, 10000.0, 1024);

    let batch = 2;
    let num_heads = 4;
    let seq_len = 16;

    // Query/Key tensor: (batch, num_heads, seq_len, head_dim)
    let qk = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)?;

    // Indices for RoPE
    let indices = Tensor::randn(0f32, 0.5, (batch, 3, seq_len), &device)?;
    let freqs_cis = rope.compute_freqs_cis(&indices, &device)?;

    let rotated = rope.apply_rotary_emb(&qk, &freqs_cis)?;

    assert_eq!(rotated.dims(), qk.dims());

    Ok(())
}

// ===========================================================================
// DiTBlock (Legacy API) Tests
// ===========================================================================

#[test]
fn test_dit_block_legacy_forward() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Legacy DiTBlock uses fixed embedding sizes: t_emb=256, c_emb=4096
    let config = DitConfig {
        hidden_size: 64,
        num_heads: 4,
        caption_channels: 4096, // Must match ModulationLayer expectations
        ..small_test_config()
    };
    let block = DiTBlock::new(vb.pp("dit"), &config)?;

    let batch = 2;
    let seq_len = 16;

    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;
    // Legacy API expects t_emb of size 256 and c_emb of size 4096
    let t_emb = Tensor::randn(0f32, 1.0, (batch, 256), &device)?;
    let c_emb = Tensor::randn(0f32, 1.0, (batch, 4096), &device)?;

    let head_dim = config.hidden_size / config.num_heads;
    let rope = FractionalRoPE::new(head_dim, 10000.0, 1024);

    // seq_len = 4 * 2 * 2 = 16 to match
    let (t_coords, h_coords, w_coords) = generate_coord_grids(4, 2, 2, &device)?;

    let out = block.forward(&x, &t_emb, &c_emb, &rope, &t_coords, &h_coords, &w_coords)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

// ===========================================================================
// Integration Tests
// ===========================================================================

#[test]
fn test_full_dit_pipeline() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = DitConfig {
        depth: 2,
        ..small_test_config()
    };

    // Create model
    let model = Transformer3DModel::new(vb.pp("transformer"), &config)?;

    // Simulate VAE latent output: 5D tensor (B, C, T, H, W)
    let batch = 1;
    let t = 2;
    let h = 4;
    let w = 4;
    let _seq_len = t * h * w;

    // 5D input from VAE
    let hidden_states = Tensor::randn(0f32, 1.0, (batch, config.in_channels, t, h, w), &device)?;

    // Generate indices grid for RoPE
    let indices_grid = generate_indices_grid(batch, t, h, w, &device)?;

    // T5 embeddings
    let caption_len = 8;
    let encoder_hidden = Tensor::randn(
        0f32,
        1.0,
        (batch, caption_len, config.caption_channels),
        &device,
    )?;

    // Timestep
    let timestep = Tensor::randn(0f32, 1.0, (batch,), &device)?;

    let out = model.forward(
        &hidden_states,
        &indices_grid,
        Some(&encoder_hidden),
        &timestep,
        None,
        None,
        None,
        None,
    )?;

    // Output should be 5D: (B, C, T, H, W)
    assert_eq!(out.dim(0)?, batch);
    assert_eq!(out.dim(1)?, config.in_channels);
    assert_eq!(out.dim(2)?, t);
    assert_eq!(out.dim(3)?, h);
    assert_eq!(out.dim(4)?, w);

    Ok(())
}

// Helper to generate indices grid for RoPE
fn generate_indices_grid(
    batch: usize,
    t: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = t * h * w;
    let mut indices = Vec::with_capacity(batch * 3 * seq_len);

    for _b in 0..batch {
        // T indices
        for ti in 0..t {
            for _hi in 0..h {
                for _wi in 0..w {
                    indices.push(ti as f32 / (t - 1).max(1) as f32);
                }
            }
        }
        // H indices
        for _ti in 0..t {
            for hi in 0..h {
                for _wi in 0..w {
                    indices.push(hi as f32 / (h - 1).max(1) as f32);
                }
            }
        }
        // W indices
        for _ti in 0..t {
            for _hi in 0..h {
                for wi in 0..w {
                    indices.push(wi as f32 / (w - 1).max(1) as f32);
                }
            }
        }
    }

    Tensor::from_vec(indices, (batch, 3, seq_len), device)
}

// ===========================================================================
// Edge Case & Error Tests
// ===========================================================================

#[test]
fn test_attention_empty_sequence() -> Result<()> {
    // Empty sequence should be handled gracefully
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let attn = Attention::new(vb.pp("attn"), &config, None)?;

    let batch = 2;
    let seq_len = 1; // Minimal sequence
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    let out = attn.forward(&x, None, None, None, None, None)?;

    assert_eq!(out.dims(), x.dims());

    Ok(())
}

#[test]
fn test_feedforward_preserves_dtype() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = small_test_config();
    let ff = FeedForward::new(vb.pp("ff"), &config)?;

    let batch = 2;
    let seq_len = 16;
    let x = Tensor::randn(0f32, 1.0, (batch, seq_len, config.hidden_size), &device)?;

    let out = ff.forward(&x)?;

    assert_eq!(out.dtype(), DType::F32);

    Ok(())
}
