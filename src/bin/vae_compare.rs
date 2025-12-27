//! Compare VAE decode outputs (Rust) against diffusers (Python) for the same `latents.bin`.
//!
//! Typical flow:
//! 1) Run `ltx-video` to produce `output/latents.bin`
//! 2) Run this binary to produce `output/vae_compare/rust_video.bin` + preview PNG(s)
//! 3) Run `python scripts/compare_vae_latents_bin.py` to decode with diffusers and compare

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

use candle_video::latents_bin::{read_f32_tensor_with_header, write_f32_tensor_with_header};
use candle_video::vae::VaeDecoder;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Decode latents.bin with Rust VAE and dump outputs for comparison"
)]
struct Args {
    /// Path to `output/latents.bin` (u64 header + f32 data)
    #[arg(long, default_value = "output/latents.bin")]
    latents: PathBuf,

    /// Path to model safetensors (single file). Used to load `vae.decoder` weights.
    #[arg(
        long,
        default_value = "ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors"
    )]
    model: PathBuf,

    /// Output directory
    #[arg(long, default_value = "output/vae_compare")]
    out_dir: PathBuf,

    /// Timestep used for timestep conditioning (if enabled)
    #[arg(long, default_value = "0.05")]
    timestep: f64,

    /// Decode on CPU (default)
    #[arg(long, default_value_t = true)]
    cpu: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("vae_compare=info".parse()?),
        )
        .init();

    let args = Args::parse();
    let device = Device::Cpu;
    let dtype = DType::F32;

    std::fs::create_dir_all(&args.out_dir).context("create out_dir")?;

    info!("Loading latents from {:?}", args.latents);
    let (_dims, latents) = read_f32_tensor_with_header(&args.latents, &device)?;
    let latents = latents.to_dtype(dtype)?;
    info!("Latents shape: {:?}", latents.dims());

    info!("Loading VAE decoder from {:?}", args.model);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&args.model], dtype, &device)? };
    let decoder = VaeDecoder::new(vb.pp("vae.decoder"))?;

    info!("Decoding with timestep={}", args.timestep);
    let video = decoder.decode(&latents, Some(args.timestep))?;
    info!("Video shape: {:?}", video.dims());

    let out_bin = args.out_dir.join("rust_video.bin");
    write_f32_tensor_with_header(&out_bin, &video)?;
    info!("Wrote {:?}", out_bin);

    let frame0 = video.i((0, .., 0, .., ..))?.unsqueeze(0)?.unsqueeze(2)?;
    save_video_frames(&frame0, &args.out_dir, "rust_frame")?;

    Ok(())
}

fn save_video_frames(video: &Tensor, output_dir: &PathBuf, prefix: &str) -> Result<()> {
    use image::{ImageBuffer, Rgb};

    // Video shape: (B, C, T, H, W)
    let video = video.to_dtype(DType::F32)?;
    let dims = video.dims();
    if dims.len() != 5 {
        anyhow::bail!("Expected 5D video tensor, got {:?}", dims);
    }

    let num_frames = dims[2];
    let height = dims[3];
    let width = dims[4];

    info!("Saving {} frame(s) to {:?}...", num_frames, output_dir);

    for t in 0..num_frames {
        let frame = video.i((0, .., t, .., ..))?;
        let frame = ((frame.clamp(-1.0, 1.0)? + 1.0)? * 127.5)?;
        let frame = frame.to_vec3::<f32>()?;

        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);
        for (y, row) in frame[0].iter().enumerate().take(height) {
            for (x, _) in row.iter().enumerate().take(width) {
                let r = frame[0][y][x].round().clamp(0.0, 255.0) as u8;
                let g = frame[1][y][x].round().clamp(0.0, 255.0) as u8;
                let b = frame[2][y][x].round().clamp(0.0, 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        let frame_path = output_dir.join(format!("{prefix}_{t:04}.png"));
        img.save(&frame_path)?;
    }

    Ok(())
}
