//! Wan VAE small parity test - compares Rust with Python diffusers on 256x256.
//!
//! Run: cargo test verify_wan_vae_small --release --features flash-attn,cudnn -- --nocapture

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};
use std::path::Path;

const REF_FILE: &str = "gen_wan_vae_small.safetensors";
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
fn verify_wan_vae_small_decode() {
    // Skip if reference file doesn't exist
    if !Path::new(REF_FILE).exists() {
        println!("Skipping test: {} not found", REF_FILE);
        println!("Run: python scripts/gen_wan_vae_small.py");
        return;
    }

    // Skip if VAE weights don't exist
    if !Path::new(VAE_PATH).exists() {
        println!("Skipping test: {} not found", VAE_PATH);
        return;
    }

    // Load reference data
    let input_latents = load_reference_tensor("input_latents").expect("input_latents not found");
    let ref_full_decode = load_reference_tensor("full_decode").expect("full_decode not found");

    println!("Input latents shape: {:?}", input_latents.dims());
    println!("Reference full decode shape: {:?}", ref_full_decode.dims());

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
    println!("\n=== Running VAE decode (256x256) ===");
    let rust_output = vae.decode(&input_latents).expect("VAE decode failed");
    println!("Rust output shape: {:?}", rust_output.dims());

    // Move to CPU for comparison
    let rust_output = rust_output.to_device(&Device::Cpu).unwrap();

    // Compare shapes
    assert_eq!(
        rust_output.dims(),
        ref_full_decode.dims(),
        "Shape mismatch: Rust {:?} vs Python {:?}",
        rust_output.dims(),
        ref_full_decode.dims()
    );

    // Compute differences
    let max_diff = compute_max_diff(&rust_output, &ref_full_decode);
    let mean_diff = compute_mean_diff(&rust_output, &ref_full_decode);

    println!("\n=== VAE Decode Parity Results (256x256) ===");
    println!("Max diff:  {:.6}", max_diff);
    println!("Mean diff: {:.6}", mean_diff);

    // BF16 tolerance
    let tolerance = 0.1;

    if max_diff > tolerance {
        println!("\n!!! PARITY FAILED !!!");
        println!("Max diff {} exceeds tolerance {}", max_diff, tolerance);

        // Print sample values
        let rust_flat = rust_output.flatten_all().unwrap();
        let ref_flat = ref_full_decode.flatten_all().unwrap();

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

    println!("\n✓ VAE decode parity check PASSED (256x256)");
}

#[test]
fn verify_wan_vae_small_shapes() {
    if !Path::new(REF_FILE).exists() {
        println!("Skipping test: {} not found", REF_FILE);
        return;
    }

    println!("\n=== Python Decoder Intermediate Shapes (256x256, Frame 0) ===");

    let shape_checks = [
        ("frame0_decoder_conv_in", vec![1, 384, 1, 32, 32]),
        ("frame0_decoder_mid_block", vec![1, 384, 1, 32, 32]),
        ("frame0_decoder_up_block_0", vec![1, 192, 1, 64, 64]),
        ("frame0_decoder_up_block_1", vec![1, 192, 1, 128, 128]),
        ("frame0_decoder_up_block_2", vec![1, 96, 1, 256, 256]),
        ("frame0_decoder_up_block_3", vec![1, 96, 1, 256, 256]),
        ("frame0_decoder_norm_out", vec![1, 96, 1, 256, 256]),
        ("frame0_decoder_conv_out", vec![1, 3, 1, 256, 256]),
    ];

    for (name, expected_shape) in shape_checks {
        if let Some(tensor) = load_reference_tensor(name) {
            let actual_shape: Vec<usize> = tensor.dims().to_vec();
            println!("{}: {:?}", name, actual_shape);
            assert_eq!(actual_shape, expected_shape, "Shape mismatch for {}", name);
        } else {
            println!("{}: NOT FOUND", name);
        }
    }

    println!("\n✓ All intermediate shapes match expected values");
}
