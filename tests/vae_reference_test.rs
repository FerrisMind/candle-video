//! VAE reference test - compare Rust output with Python diffusers reference
//!
//! Run: cargo test --release --test vae_reference_test -- --ignored --nocapture
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_vae_parity.rs for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_video::vae::VaeDecoder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load binary tensor from file
fn load_bin_tensor(path: &Path, shape: &[usize], device: &Device) -> Result<Tensor> {
    let mut file = File::open(path).expect("Failed to open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    let len = buffer.len() / 4;
    let floats: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(floats.len(), len);
    Tensor::from_vec(floats, shape, device)
}

#[test]
#[ignore = "Requires model weights and reference data"]
fn test_vae_decoder_model_loads() -> Result<()> {
    let device = Device::Cpu;

    // Load model weights
    let model_path = "ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return Ok(());
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let vb_decoder = vb.pp("vae.decoder"); // Note: vae.decoder, not decoder

    println!("Loading VAE decoder...");
    let decoder = VaeDecoder::new(vb_decoder)?;
    println!("✓ VAE decoder loaded successfully!");

    // Simple forward pass with random input
    let latents = Tensor::randn(0f32, 1.0, (1, 128, 1, 4, 6), &device)?;
    let output = decoder.decode(&latents, Some(0.05))?;

    println!("  Input:  {:?}", latents.shape());
    println!("  Output: {:?}", output.shape());

    assert_eq!(output.dim(0)?, 1, "batch size");
    assert_eq!(output.dim(1)?, 3, "RGB channels");

    println!("✓ Forward pass completed!");
    Ok(())
}

#[test]
#[ignore = "Requires model weights and reference data"]
fn test_vae_matches_diffusers_reference() -> Result<()> {
    println!("Loading reference data...");

    let device = Device::Cpu;

    // Load reference data
    let latents_path = Path::new("output/debug/vae_latents.bin");
    let output_path = Path::new("output/debug/vae_output_ref.bin");

    if !latents_path.exists() || !output_path.exists() {
        eprintln!("Reference data not found. Run: python scripts/compare_vae.py");
        return Ok(());
    }

    // Load latents: (1, 128, 1, 4, 6)
    let latents = load_bin_tensor(latents_path, &[1, 128, 1, 4, 6], &device)?;
    println!("  Latents: {:?}", latents.shape());

    // Load reference output: (1, 3, 1, 128, 192)
    let reference = load_bin_tensor(output_path, &[1, 3, 1, 128, 192], &device)?;
    println!("  Reference: {:?}", reference.shape());

    // Load model
    println!("\nLoading VAE decoder...");
    let model_path = "ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found at {}", model_path);
        return Ok(());
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let vb_decoder = vb.pp("vae.decoder");
    let decoder = VaeDecoder::new(vb_decoder)?;
    println!("  Decoder loaded!");

    // Run Rust forward pass
    println!("\nRunning Rust forward pass...");
    let output = decoder.decode(&latents, Some(0.05))?;
    println!("  Rust output: {:?}", output.shape());

    // Compare
    println!("\nComparison:");
    let diff = (&output - &reference)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    println!("  Max diff: {:.6}", max_diff);
    println!("  Mean diff: {:.6}", mean_diff);

    let tolerance = 1e-3;
    if max_diff < tolerance {
        println!("\n✓ PASS: Rust output matches diffusers reference!");
    } else {
        eprintln!(
            "\n✗ FAIL: max_diff {} exceeds tolerance {}",
            max_diff, tolerance
        );

        // Debug info
        let rust_min = output.min_all()?.to_scalar::<f32>()?;
        let rust_max = output.max_all()?.to_scalar::<f32>()?;
        let ref_min = reference.min_all()?.to_scalar::<f32>()?;
        let ref_max = reference.max_all()?.to_scalar::<f32>()?;

        println!("\n  Rust output range: [{:.4}, {:.4}]", rust_min, rust_max);
        println!("  Reference range:   [{:.4}, {:.4}]", ref_min, ref_max);
    }

    assert!(
        max_diff < tolerance,
        "VAE output differs from reference: max_diff = {}",
        max_diff
    );

    Ok(())
}
