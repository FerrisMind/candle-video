//! Debug test for DiT intermediate values
//! Run with: cargo test --test dit_debug_test -- --nocapture
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_dit_parity.rs for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_video::{config::DitConfig, dit::Transformer3DModel, rope::generate_indices_grid};
use std::fs::File;
use std::io::Read;

fn load_f32_binary(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect(&format!("Failed to open {}", path));
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    buffer
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
#[ignore] // Requires external binary files and model weights
fn test_dit_debug_values() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load test inputs created by Python
    let latents_data = load_f32_binary("output/dit_test_latents.bin");
    let text_emb_data = load_f32_binary("output/dit_test_text_emb.bin");

    // Create tensors: (B=1, C=128, T=1, H=16, W=24)
    let latents = Tensor::from_vec(latents_data, (1, 128, 1, 16, 24), &device)?.to_dtype(dtype)?;

    // Text: (B=1, seq=6, dim=4096)
    let text_emb = Tensor::from_vec(text_emb_data, (1, 6, 4096), &device)?.to_dtype(dtype)?;

    // Timestep
    let timestep = Tensor::new(&[1.0f32], &device)?.to_dtype(dtype)?;

    // Print input stats
    let lat_flat = latents.flatten_all()?;
    println!("\n=== Rust DiT Debug ===");
    println!("Input latents: {:?}", latents.shape());
    println!(
        "  range: [{:.4}, {:.4}]",
        lat_flat.min(0)?.to_scalar::<f32>()?,
        lat_flat.max(0)?.to_scalar::<f32>()?
    );

    // Load model
    let model_path = "ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors";
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
    let vb = vb.pp("model.diffusion_model");

    let config = DitConfig {
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
    };

    let model = Transformer3DModel::new(vb, &config)?;
    println!("Model loaded");

    // Generate indices grid
    let indices_grid = generate_indices_grid(1, 1, 16, 24, &device)?;

    // Run forward pass
    let output = model.forward(
        &latents,
        &indices_grid,
        Some(&text_emb),
        &timestep,
        None,
        None,
        None,
        None,
    )?;

    let out_flat = output.flatten_all()?;
    println!("\nOutput: {:?}", output.shape());
    println!(
        "  range: [{:.4}, {:.4}]",
        out_flat.min(0)?.to_scalar::<f32>()?,
        out_flat.max(0)?.to_scalar::<f32>()?
    );
    println!("  mean: {:.6}", out_flat.mean_all()?.to_scalar::<f32>()?);

    // Expected from Python: range [-3.4189, 3.8338]
    println!("\nExpected (Python): range [-3.4189, 3.8338], mean ~0.04");

    Ok(())
}
