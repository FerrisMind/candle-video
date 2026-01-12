//! Test Wan VAE decode at 480x480 resolution
//!
//! Run: cargo run --example test_wan_vae_480 --release

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};
use std::path::Path;

const VAE_PATH: &str = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors";

fn main() -> anyhow::Result<()> {
    println!("=== Wan VAE 480x480 Memory Test ===\n");

    if !Path::new(VAE_PATH).exists() {
        println!("VAE not found at {}", VAE_PATH);
        return Ok(());
    }

    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    // Load VAE
    println!("\nLoading VAE...");
    let config = AutoencoderKLWanConfig::wan_2_1();
    let vae = load_vae(VAE_PATH, config, &device, DType::BF16)?;
    println!("VAE loaded");

    // Test 480x480 with 33 frames -> latent: 9x60x60
    let num_frames = 33;
    let height = 480;
    let width = 480;
    let latent_frames = (num_frames - 1) / 4 + 1; // 9
    let latent_h = height / 8; // 60
    let latent_w = width / 8; // 60

    println!("\nTest case: {} frames × {}×{}", num_frames, height, width);
    println!("Latent shape: [1, 16, {}, {}, {}]", latent_frames, latent_h, latent_w);

    // Create deterministic input
    let latents = Tensor::randn(0f32, 1f32, (1, 16, latent_frames, latent_h, latent_w), &device)?
        .to_dtype(DType::BF16)?;

    println!("\nRunning VAE decode...");
    
    // Try to decode
    match vae.decode(&latents) {
        Ok(output) => {
            println!("Success! Output shape: {:?}", output.dims());
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    println!("\nDone.");
    Ok(())
}
