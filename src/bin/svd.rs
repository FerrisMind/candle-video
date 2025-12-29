//! SVD CLI - Stable Video Diffusion Command Line Interface
//!
//! Generate video from an input image using Stable Video Diffusion.
//! Fixed output: 14 frames at 576x1024 resolution.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use std::path::Path;

use candle_video::svd::{
    SvdConfig, SvdInferenceConfig, SvdPipeline, load_image, save_video_frames,
};

// SVD fixed parameters
const NUM_FRAMES: usize = 14;
const HEIGHT: usize = 576;
const WIDTH: usize = 1024;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate video from image using Stable Video Diffusion (14 frames @ 576x1024)"
)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    image: String,

    /// Output directory for video frames
    #[arg(short, long, default_value = "output")]
    output: String,

    /// Path to model directory (containing unet/, vae/, image_encoder/)
    #[arg(short, long)]
    model: String,

    /// Number of inference steps
    #[arg(long, default_value_t = 25)]
    steps: usize,

    /// Frames per second
    #[arg(long, default_value_t = 7)]
    fps: usize,

    /// Motion bucket ID (0-255, higher = more motion)
    #[arg(long, default_value_t = 127)]
    motion_bucket_id: usize,

    /// Noise augmentation strength
    #[arg(long, default_value_t = 0.02)]
    noise_aug_strength: f64,

    /// Minimum guidance scale
    #[arg(long, default_value_t = 1.0)]
    min_guidance: f64,

    /// Maximum guidance scale
    #[arg(long, default_value_t = 3.0)]
    max_guidance: f64,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,
}

struct ModelPaths {
    unet: std::path::PathBuf,
    vae: std::path::PathBuf,
    clip: std::path::PathBuf,
}

fn find_model_paths(model_dir: &Path, prefer_fp16: bool) -> Result<(ModelPaths, bool)> {
    // Try FP16 first if preferred, then fallback to FP32

    // UNet
    let unet_fp16 = model_dir.join("unet/diffusion_pytorch_model.fp16.safetensors");
    let unet_fp32 = model_dir.join("unet/diffusion_pytorch_model.safetensors");
    let (unet, unet_is_fp16) = if prefer_fp16 && unet_fp16.exists() {
        (unet_fp16, true)
    } else {
        (unet_fp32, false)
    };

    // VAE
    let vae_fp16 = model_dir.join("vae/diffusion_pytorch_model.fp16.safetensors");
    let vae_fp32 = model_dir.join("vae/diffusion_pytorch_model.safetensors");
    let vae = if vae_fp16.exists() {
        vae_fp16
    } else {
        vae_fp32
    };

    // CLIP
    let clip_fp16 = model_dir.join("image_encoder/model.fp16.safetensors");
    let clip_fp32 = model_dir.join("image_encoder/model.safetensors");
    let clip = if clip_fp16.exists() {
        clip_fp16
    } else {
        clip_fp32
    };

    // Verify files exist
    if !unet.exists() {
        anyhow::bail!("UNet weights not found: {}", unet.display());
    }
    if !vae.exists() {
        anyhow::bail!("VAE weights not found: {}", vae.display());
    }
    if !clip.exists() {
        anyhow::bail!("CLIP weights not found: {}", clip.display());
    }

    Ok((ModelPaths { unet, vae, clip }, unet_is_fp16))
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🎬 SVD - Stable Video Diffusion");
    println!("================================");
    println!("Input image: {}", args.image);
    println!("Output directory: {}", args.output);
    println!("Model directory: {}", args.model);
    println!("Output: {} frames @ {}x{}", NUM_FRAMES, WIDTH, HEIGHT);
    println!("Steps: {}, FPS: {}", args.steps, args.fps);
    println!(
        "Motion bucket: {}, Noise aug: {}",
        args.motion_bucket_id, args.noise_aug_strength
    );
    println!("Guidance: {} -> {}", args.min_guidance, args.max_guidance);
    println!();

    // Setup device
    let device = if args.cpu {
        println!("Using CPU");
        Device::Cpu
    } else {
        println!("Using CUDA");
        Device::cuda_if_available(0)?
    };

    // Find model paths (prefer FP16 on GPU, FP32 on CPU)
    println!("\n📦 Loading model weights...");
    let model_dir = Path::new(&args.model);
    // Always prefer FP16 if available, as requested by user. 
    // Even on CPU, we'll try to load FP16 (though running it might be slow or require f32 cast if ops are missing).
    let prefer_fp16 = true; 
    let (paths, is_fp16) = find_model_paths(model_dir, prefer_fp16)?;

    let dtype = if is_fp16 { DType::F16 } else { DType::F32 };
    println!(
        "Dtype: {:?} ({})",
        dtype,
        if args.cpu {
            "CPU mode"
        } else {
            "auto-detected"
        }
    );

    println!("  UNet: {}", paths.unet.display());
    println!("  VAE: {}", paths.vae.display());
    println!("  CLIP: {}", paths.clip.display());

    // Load each component separately
    let unet_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&paths.unet], dtype, &device)? };
    let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&paths.vae], dtype, &device)? };
    let clip_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&paths.clip], dtype, &device)? };

    // Create config
    let mut config = SvdConfig::default();
    
    // Disable VAE force upcast if running in F16 to avoid dtype mismatch
    // This allows the VAE to process F16 inputs with F16 weights
    if dtype == DType::F16 {
        config.vae.force_upcast = false;
    }

    // Create inference config with fixed resolution
    let inference_config = SvdInferenceConfig {
        num_frames: NUM_FRAMES,
        height: HEIGHT,
        width: WIDTH,
        num_inference_steps: args.steps,
        min_guidance_scale: args.min_guidance,
        max_guidance_scale: args.max_guidance,
        fps: args.fps,
        motion_bucket_id: args.motion_bucket_id,
        noise_aug_strength: args.noise_aug_strength,
        seed: args.seed,
        ..Default::default() // Use default decode_chunk_size
    };

    // Create pipeline
    println!("🔧 Creating pipeline...");
    let mut pipeline =
        SvdPipeline::new_from_parts(unet_vb, vae_vb, clip_vb, &config, device.clone(), dtype)
            .context("Failed to create pipeline")?;

    // Load input image
    println!("🖼️  Loading input image...");
    let image = load_image(&args.image, HEIGHT, WIDTH, &device, dtype)
        .context("Failed to load input image")?;

    // Generate video
    println!("🎥 Generating video ({} frames)...", NUM_FRAMES);
    let start = std::time::Instant::now();

    let video_frames = pipeline
        .generate(&image, &inference_config)
        .context("Failed to generate video")?;

    let elapsed = start.elapsed();
    println!("✅ Generation complete in {:.2}s", elapsed.as_secs_f64());
    println!(
        "   ({:.2} frames/sec)",
        NUM_FRAMES as f64 / elapsed.as_secs_f64()
    );

    // Save frames
    println!("\n💾 Saving frames to {}...", args.output);
    save_video_frames(&video_frames, &args.output).context("Failed to save video frames")?;

    println!("\n🎉 Done! Video frames saved to {}/", args.output);
    println!("   Use ffmpeg to create video:");
    println!(
        "   ffmpeg -framerate {} -i {}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4",
        args.fps, args.output
    );

    Ok(())
}
