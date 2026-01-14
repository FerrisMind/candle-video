//! Wan VAE parity test - compares Rust implementation with Python diffusers.
//!
//! Run: cargo test verify_wan_vae_parity --release -- --nocapture

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};
use std::path::Path;

const REF_FILE: &str = "gen_wan_vae_ref.safetensors";
const VAE_PATH: &str = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors";

fn load_reference_tensor(name: &str) -> Option<Tensor> {
    if !Path::new(REF_FILE).exists() {
        return None;
    }
    let tensors = candle_core::safetensors::load(REF_FILE, &Device::Cpu).ok()?;
    tensors.get(name).cloned()
}

fn compute_max_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let diff = a.sub(&b).unwrap().abs().unwrap();
    diff.max(0).unwrap().to_scalar::<f32>().unwrap()
}

fn compute_mean_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let diff = a.sub(&b).unwrap().abs().unwrap();
    diff.mean_all().unwrap().to_scalar::<f32>().unwrap()
}

#[test]
fn verify_wan_vae_decode_parity() {
    // Skip if reference file doesn't exist
    if !Path::new(REF_FILE).exists() {
        println!("Skipping test: {} not found", REF_FILE);
        println!("Run: python scripts/gen_wan_vae_ref.py");
        return;
    }

    // Skip if VAE weights don't exist
    if !Path::new(VAE_PATH).exists() {
        println!("Skipping test: {} not found", VAE_PATH);
        return;
    }

    // Load reference data
    let input_latents = load_reference_tensor("input_latents").expect("input_latents not found");
    let ref_output = load_reference_tensor("full_decode").expect("full_decode not found");

    println!("Input latents shape: {:?}", input_latents.dims());
    println!("Reference output shape: {:?}", ref_output.dims());

    // Set up device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    // Load VAE
    let config = AutoencoderKLWanConfig::wan_2_1();
    let vae = load_vae(VAE_PATH, config, &device, DType::BF16).expect("Failed to load VAE");
    println!("VAE loaded");

    // Move input to device and convert to BF16
    let input_latents = input_latents
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Run decode
    println!("\nRunning VAE decode...");
    let rust_output = vae.decode(&input_latents).expect("VAE decode failed");
    println!("Rust output shape: {:?}", rust_output.dims());

    // Move to CPU for comparison
    let rust_output = rust_output.to_device(&Device::Cpu).unwrap();

    // Compare shapes
    assert_eq!(
        rust_output.dims(),
        ref_output.dims(),
        "Shape mismatch: Rust {:?} vs Python {:?}",
        rust_output.dims(),
        ref_output.dims()
    );

    // Compute differences
    let max_diff = compute_max_diff(&rust_output, &ref_output);
    let mean_diff = compute_mean_diff(&rust_output, &ref_output);

    println!("\n=== VAE Decode Parity Results ===");
    println!("Max diff:  {:.6}", max_diff);
    println!("Mean diff: {:.6}", mean_diff);

    // BF16 tolerance - allow some numerical difference
    let tolerance = 0.1; // BF16 has limited precision

    if max_diff > tolerance {
        println!("\n!!! PARITY FAILED !!!");
        println!("Max diff {} exceeds tolerance {}", max_diff, tolerance);

        // Print sample values for debugging
        let rust_flat = rust_output.flatten_all().unwrap();
        let ref_flat = ref_output.flatten_all().unwrap();

        println!("\nSample values (first 10):");
        for i in 0..10.min(rust_flat.dims1().unwrap()) {
            let r = rust_flat.get(i).unwrap().to_scalar::<f32>().unwrap();
            let p = ref_flat.get(i).unwrap().to_scalar::<f32>().unwrap();
            println!(
                "  [{}] Rust: {:.6}, Python: {:.6}, diff: {:.6}",
                i,
                r,
                p,
                (r - p).abs()
            );
        }

        panic!("VAE decode parity check failed");
    }

    println!("\n✓ VAE decode parity check PASSED");
}

#[test]
fn verify_wan_vae_post_quant_conv_parity() {
    // Skip if reference file doesn't exist
    if !Path::new(REF_FILE).exists() {
        println!("Skipping test: {} not found", REF_FILE);
        return;
    }

    if !Path::new(VAE_PATH).exists() {
        println!("Skipping test: {} not found", VAE_PATH);
        return;
    }

    let input_latents = load_reference_tensor("input_latents").expect("input_latents not found");
    let ref_post_quant =
        load_reference_tensor("after_post_quant_conv").expect("after_post_quant_conv not found");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    let config = AutoencoderKLWanConfig::wan_2_1();
    let vae = load_vae(VAE_PATH, config, &device, DType::BF16).expect("Failed to load VAE");

    let input_latents = input_latents
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Run post_quant_conv only
    let rust_post_quant = vae
        .post_quant_conv
        .forward(&input_latents)
        .expect("post_quant_conv failed");
    let rust_post_quant = rust_post_quant.to_device(&Device::Cpu).unwrap();

    let max_diff = compute_max_diff(&rust_post_quant, &ref_post_quant);
    let mean_diff = compute_mean_diff(&rust_post_quant, &ref_post_quant);

    println!("\n=== Post Quant Conv Parity Results ===");
    println!("Max diff:  {:.6}", max_diff);
    println!("Mean diff: {:.6}", mean_diff);

    let tolerance = 0.01;
    assert!(
        max_diff < tolerance,
        "Post quant conv parity failed: max_diff {} > tolerance {}",
        max_diff,
        tolerance
    );

    println!("✓ Post quant conv parity check PASSED");
}
