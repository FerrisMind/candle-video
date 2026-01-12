//! Debug Wan VAE memory usage step by step.
//!
//! Run: cargo run --example debug_wan_vae_mem --release --features flash-attn,cudnn

use candle_core::{DType, Device, Tensor};
use candle_core::backend::BackendDevice;
use candle_video::models::wan::{AutoencoderKLWanConfig, load_vae};
use std::process::Command;

fn get_gpu_memory_mb() -> f64 {
    // Use nvidia-smi to get actual GPU memory
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    
    match output {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            // Take first line (first GPU)
            s.lines().next().unwrap_or("0").trim().parse::<f64>().unwrap_or(0.0)
        }
        Err(_) => 0.0,
    }
}

fn main() -> anyhow::Result<()> {
    println!("=== Wan VAE Memory Debug (Rust) ===\n");
    
    let vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors";
    
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);
    println!("Initial GPU memory: {:.0} MB ({:.3} GB)", get_gpu_memory_mb(), get_gpu_memory_mb() / 1024.0);
    
    // Load VAE
    println!("\nLoading VAE...");
    let config = AutoencoderKLWanConfig::wan_2_1();
    let vae = load_vae(vae_path, config, &device, DType::BF16)?;
    println!("VAE loaded. GPU memory: {:.0} MB ({:.3} GB)", get_gpu_memory_mb(), get_gpu_memory_mb() / 1024.0);
    
    // Test 256x256 first (known to work)
    println!("\n--- Test 256x256 x 17 frames ---");
    let latent_frames = 5;  // (17-1)/4 + 1
    let latent_h = 32;      // 256/8
    let latent_w = 32;
    
    println!("Creating latents [1, 16, {}, {}, {}]...", latent_frames, latent_h, latent_w);
    let latents = Tensor::randn(0f32, 1f32, (1, 16, latent_frames, latent_h, latent_w), &device)?
        .to_dtype(DType::BF16)?;
    println!("After creating latents: {:.0} MB ({:.3} GB)", get_gpu_memory_mb(), get_gpu_memory_mb() / 1024.0);
    
    println!("Running VAE decode...");
    let output = vae.decode(&latents)?;
    println!("Output shape: {:?}", output.dims());
    println!("After decode: {:.0} MB ({:.3} GB)", get_gpu_memory_mb(), get_gpu_memory_mb() / 1024.0);
    
    // Drop output
    drop(output);
    drop(latents);
    
    // Force sync
    if let Device::Cuda(cuda_dev) = &device {
        cuda_dev.synchronize()?;
    }
    
    println!("After dropping tensors: {:.0} MB ({:.3} GB)", get_gpu_memory_mb(), get_gpu_memory_mb() / 1024.0);
    
    Ok(())
}
