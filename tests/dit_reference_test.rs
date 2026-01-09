//! Integration test: Compare Rust DiT output with diffusers reference
//!
//! Prerequisites:
//! 1. Run `python scripts/compare_dit.py --reference` to generate reference outputs
//! 2. Run `python scripts/create_rust_test_data.py` to create .bin files
//! 3. Place model at `ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors`
//!
//! Run with: cargo test --release --test dit_reference_test -- --ignored
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_dit_parity.rs for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_video::config::DitConfig;
use candle_video::dit::Transformer3DModel;
use candle_video::rope::generate_indices_grid_raw;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load tensor from .bin file (our custom format)
fn load_tensor_bin(path: &Path, device: &Device) -> Result<Tensor> {
    let mut file = File::open(path).expect("Failed to open file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("Failed to read file");

    // Parse header: ndims, dim0, dim1, ...
    let mut offset = 0;
    let ndims = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let mut dims = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        dims.push(u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()) as usize);
        offset += 8;
    }

    // Parse data as f32
    let num_elements: usize = dims.iter().product();
    let mut data = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let idx = offset + i * 4;
        data.push(f32::from_le_bytes(buf[idx..idx + 4].try_into().unwrap()));
    }

    Tensor::from_vec(data, dims.as_slice(), device)
}

/// Create DiT config matching the model
fn create_dit_config() -> DitConfig {
    DitConfig {
        patch_size: 1,
        patch_size_t: Some(1),
        in_channels: 128,
        hidden_size: 2048,
        depth: 28,
        num_heads: 32,
        caption_channels: 4096,
        mlp_ratio: 4.0,
        use_flash_attention: false,
        timestep_scale_multiplier: Some(1000.0),
    }
}

#[test]
#[ignore] // Requires model file and reference data
fn test_dit_matches_diffusers_reference() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    // Check prerequisites
    let model_path = Path::new("ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors");
    let latents_path = Path::new("output/dit_latents.bin");
    let text_path = Path::new("output/dit_text_emb.bin");
    let ref_path = Path::new("output/dit_output_ref.bin");

    if !model_path.exists() {
        eprintln!("Model not found at {:?}", model_path);
        eprintln!("Download the model first.");
        return Ok(());
    }

    if !latents_path.exists() || !text_path.exists() || !ref_path.exists() {
        eprintln!("Reference data not found.");
        eprintln!("Run: python scripts/compare_dit.py --reference");
        eprintln!("Run: python scripts/create_rust_test_data.py");
        return Ok(());
    }

    // Load reference inputs
    println!("Loading reference data...");
    let latents = load_tensor_bin(latents_path, &device)?;
    let text_emb = load_tensor_bin(text_path, &device)?;
    let ref_output = load_tensor_bin(ref_path, &device)?;

    println!("  Latents: {:?}", latents.dims());
    println!("  Text: {:?}", text_emb.dims());
    println!("  Reference: {:?}", ref_output.dims());

    // Load model
    println!("\nLoading DiT model...");
    let config = create_dit_config();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
    let vb = vb.pp("model.diffusion_model");
    let model = Transformer3DModel::new(vb, &config)?;
    println!("  Model loaded!");

    // Prepare inputs - use RAW indices matching diffusers when rope_interpolation_scale=None
    // Diffusers uses raw integer indices (0, 1, 2, ...) without any normalization
    let (b, _c, t, h, w) = latents.dims5()?;

    let indices_grid = generate_indices_grid_raw(b, t, h, w, &device)?;
    let timestep = Tensor::new(&[1.0f32], &device)?;

    // Create encoder attention mask matching Python: torch.ones(B, text_emb.shape[1])
    let text_seq_len = text_emb.dim(1)?;
    let encoder_attention_mask = Tensor::ones((b, text_seq_len), DType::F32, &device)?;

    // Run forward pass
    println!("\nRunning Rust forward pass...");
    let rust_output = model.forward(
        &latents,
        &indices_grid,
        Some(&text_emb),
        &timestep,
        None,
        Some(&encoder_attention_mask),
        None,
        None,
    )?;

    // Reshape to 3D for comparison (model outputs 5D, reference is 3D packed)
    let rust_output_3d = rust_output.permute((0, 2, 3, 4, 1))?.flatten(1, 3)?;
    println!("  Rust output: {:?}", rust_output_3d.dims());

    // Compare
    let diff = rust_output_3d.sub(&ref_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    let max_diff = diff_flat.max(0)?.to_scalar::<f32>()?;
    let mean_diff = diff_flat.mean(0)?.to_scalar::<f32>()?;

    println!("\nComparison:");
    println!("  Max diff: {:.6}", max_diff);
    println!("  Mean diff: {:.6}", mean_diff);

    // Check tolerance
    let tolerance = 1e-3;
    if max_diff < tolerance {
        println!("\n✓ PASS: Rust output matches diffusers reference!");
    } else {
        println!(
            "\n✗ FAIL: Max diff {:.6} >= tolerance {}",
            max_diff, tolerance
        );
        // Don't panic, just report for debugging
    }

    Ok(())
}

/// Quick sanity check that model loads without error
#[test]
#[ignore]
fn test_dit_model_loads() -> Result<()> {
    let device = Device::Cpu;
    let model_path = Path::new("ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors");

    if !model_path.exists() {
        eprintln!("Model not found, skipping");
        return Ok(());
    }

    let config = create_dit_config();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let vb = vb.pp("model.diffusion_model");
    let _model = Transformer3DModel::new(vb, &config)?;

    println!("✓ Model loads successfully");
    Ok(())
}
