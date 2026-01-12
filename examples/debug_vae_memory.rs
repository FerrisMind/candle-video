// Debug VAE memory usage step by step
//
// Run with:
//   cargo run --example debug_vae_memory --release --features flash-attn,cudnn

use candle_core::{DType, Device, Tensor};
use std::path::Path;

fn get_cuda_memory_mb() -> f64 {
    // Note: Candle doesn't have a direct API for this
    // We'll print tensor sizes instead
    0.0
}

fn main() -> anyhow::Result<()> {
    println!("======================================================================");
    println!("Wan VAE Debug - Rust Implementation");
    println!("======================================================================");

    let vae_path = Path::new("models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors");
    if !vae_path.exists() {
        println!("VAE not found at {:?}", vae_path);
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    println!("Device: CUDA");
    println!("Dtype: BF16");

    // Load VAE
    println!("\n======================================================================");
    println!("Loading VAE...");

    use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};
    let config = AutoencoderKLWanConfig::wan_2_1();
    let vae = load_vae(vae_path, config, &device, dtype)?;

    println!("VAE loaded");
    println!("  decoder_conv_num: {}", 50); // hardcoded in vae.rs

    // Test case: 33 frames at 480x480 (latent: 9x60x60)
    let num_frames = 33usize;
    let height = 480usize;
    let width = 480usize;
    let latent_frames = (num_frames - 1) / 4 + 1; // 9
    let latent_h = height / 8; // 60
    let latent_w = width / 8; // 60

    println!("\n======================================================================");
    println!("Test: {} frames × {}×{}", num_frames, height, width);
    println!("Latent: {} × {} × {}", latent_frames, latent_h, latent_w);

    // Create test latents
    let latents = Tensor::randn(
        0f32,
        1f32,
        (1, 16, latent_frames, latent_h, latent_w),
        &device,
    )?
    .to_dtype(dtype)?;

    println!("\nLatents created: {:?}", latents.dims());
    println!(
        "Latents size: {:.3} MB",
        latents.elem_count() as f64 * 2.0 / 1024.0 / 1024.0
    );

    // Denormalize
    println!("\n======================================================================");
    println!("Step 1: Denormalize latents");
    let denormalized = vae.denormalize_latents(&latents)?;
    println!("  Output: {:?}", denormalized.dims());

    // Decode
    println!("\n======================================================================");
    println!("Step 2: Decode");
    println!("  Starting decode...");

    // Synchronize before decode
    device.synchronize()?;

    let decoded = vae.decode(&denormalized)?;

    // Synchronize after decode
    device.synchronize()?;

    println!("  Output: {:?}", decoded.dims());
    println!(
        "  Output size: {:.3} MB",
        decoded.elem_count() as f64 * 2.0 / 1024.0 / 1024.0
    );

    println!("\n======================================================================");
    println!("Decode successful!");

    Ok(())
}
