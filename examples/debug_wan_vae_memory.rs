//! Debug Wan VAE memory usage step by step.
//!
//! Run: cargo run --example debug_wan_vae_memory --release --features flash-attn,cudnn

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};
use std::path::Path;

const VAE_PATH: &str = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors";
const REF_FILE: &str = "gen_wan_vae_decoder_ref.safetensors";

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(70));
    println!("Wan VAE Memory Debug");
    println!("{}", "=".repeat(70));

    // Check files exist
    if !Path::new(VAE_PATH).exists() {
        println!("VAE not found at {}", VAE_PATH);
        return Ok(());
    }

    // Set up device
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    // Load VAE
    println!("\nLoading VAE...");
    let config = AutoencoderKLWanConfig::wan_2_1();
    let vae = load_vae(VAE_PATH, config, &device, DType::BF16)?;
    println!("VAE loaded");

    // Test case: 480x480 with 33 frames (latent: 9x60x60)
    let latent_frames = 9;
    let latent_h = 60;
    let latent_w = 60;

    println!(
        "\nTest case: latent shape [1, 16, {}, {}, {}]",
        latent_frames, latent_h, latent_w
    );

    // Load reference input or create random
    let latents = if Path::new(REF_FILE).exists() {
        println!("Loading reference latents from {}", REF_FILE);
        let tensors = candle_core::safetensors::load(REF_FILE, &Device::Cpu)?;
        tensors
            .get("input_latents")
            .ok_or_else(|| anyhow::anyhow!("input_latents not found"))?
            .to_device(&device)?
            .to_dtype(DType::BF16)?
    } else {
        println!("Creating random latents (seed=42)");
        Tensor::randn(
            0f32,
            1f32,
            (1, 16, latent_frames, latent_h, latent_w),
            &device,
        )?
        .to_dtype(DType::BF16)?
    };

    println!("Latents shape: {:?}", latents.dims());

    println!("\n{}", "=".repeat(70));
    println!("Running VAE decode...");

    // Denormalize
    let denormalized = vae.denormalize_latents(&latents)?;
    println!("Denormalized shape: {:?}", denormalized.dims());

    // Decode
    let output = vae.decode(&denormalized)?;
    println!("Output shape: {:?}", output.dims());

    println!("\n✓ VAE decode completed successfully!");

    // Compare with reference if available
    if Path::new(REF_FILE).exists() {
        println!("\n{}", "=".repeat(70));
        println!("Comparing with Python reference...");

        let tensors = candle_core::safetensors::load(REF_FILE, &Device::Cpu)?;
        let ref_output = tensors
            .get("final_output")
            .ok_or_else(|| anyhow::anyhow!("final_output not found"))?;

        let rust_output = output.to_device(&Device::Cpu)?;

        // Check shapes match
        if rust_output.dims() != ref_output.dims() {
            println!(
                "Shape mismatch: Rust {:?} vs Python {:?}",
                rust_output.dims(),
                ref_output.dims()
            );
        } else {
            // Compute diff
            let rust_f32 = rust_output.to_dtype(DType::F32)?.flatten_all()?;
            let ref_f32 = ref_output.to_dtype(DType::F32)?.flatten_all()?;
            let diff = rust_f32.sub(&ref_f32)?.abs()?;
            let max_diff = diff.max(0)?.to_scalar::<f32>()?;
            let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

            println!("Max diff:  {:.6}", max_diff);
            println!("Mean diff: {:.6}", mean_diff);

            if max_diff < 0.1 {
                println!("✓ Parity check PASSED");
            } else {
                println!("✗ Parity check FAILED");
            }
        }
    }

    Ok(())
}
