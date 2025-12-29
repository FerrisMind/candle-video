//! VAE Decoder Test Binary
//!
//! Decodes latents saved by debug_diffusers_latents.py script.
//! This isolates VAE decoder issues from DiT issues.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_video::vae::VaeDecoder;
use clap::Parser;
use image::{ImageBuffer, Rgb};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use tracing::{Level, info};

#[derive(Parser, Debug)]
#[command(author, version, about = "Test VAE decoder with pre-computed latents")]
struct Args {
    /// Path to latents .bin file
    #[arg(index = 1)]
    latents_path: String,

    /// Model path containing VAE weights
    #[arg(long, default_value = "ltxv-2b-0.9.8-distilled")]
    model_path: String,

    /// Output directory for decoded frames
    #[arg(long, default_value = "output/vae_test")]
    output: String,
}

fn load_latents_bin(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header: ndims
    let mut ndims_buf = [0u8; 8];
    reader.read_exact(&mut ndims_buf)?;
    let ndims = u64::from_le_bytes(ndims_buf) as usize;

    // Read shape
    let mut shape = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        let mut dim_buf = [0u8; 8];
        reader.read_exact(&mut dim_buf)?;
        shape.push(u64::from_le_bytes(dim_buf) as usize);
    }

    info!("Loading latents with shape: {:?}", shape);

    // Read data
    let num_elements: usize = shape.iter().product();
    let mut data = vec![0f32; num_elements];
    let data_bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, num_elements * 4) };
    reader.read_exact(data_bytes)?;

    let tensor = Tensor::from_vec(data, shape, device)?;
    let flat = tensor.flatten_all()?;
    let min_val: f32 = flat.min(0)?.to_scalar()?;
    let max_val: f32 = flat.max(0)?.to_scalar()?;
    info!(
        "Loaded tensor: {:?}, range: [{:.4}, {:.4}]",
        tensor.dims(),
        min_val,
        max_val,
    );

    Ok(tensor)
}

fn save_frame(video: &Tensor, frame_idx: usize, output_dir: &Path) -> Result<()> {
    // video shape: [1, C, T, H, W] -> get frame [C, H, W]
    let frame = video.i((0, .., frame_idx, .., ..))?;
    let (c, h, w) = frame.dims3()?;

    // Convert to [H, W, C] and f32
    let frame = frame.permute((1, 2, 0))?;
    let frame = frame.to_dtype(DType::F32)?;

    // Clamp to [0, 1] and scale to [0, 255]
    let frame = frame.clamp(0f32, 1f32)?.affine(255.0, 0.0)?;
    let data: Vec<f32> = frame.flatten_all()?.to_vec1()?;

    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * c;
            let r = data[idx] as u8;
            let g = data[idx + 1] as u8;
            let b = data[idx + 2] as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    let output_path = output_dir.join(format!("rust_frame_{:03}.png", frame_idx));
    img.save(&output_path)?;
    info!("Saved frame {} to {:?}", frame_idx, output_path);

    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let args = Args::parse();

    info!("VAE Decoder Test");
    info!("================");
    info!("Latents: {}", args.latents_path);
    info!("Model: {}", args.model_path);

    let device = Device::cuda_if_available(0)?;
    info!("Device: {:?}", device);

    // Load latents
    let latents = load_latents_bin(&args.latents_path, &device)?;
    let latents = latents.to_dtype(DType::BF16)?;

    // Load VAE
    info!("Loading VAE decoder...");

    let model_path = Path::new(&args.model_path);
    let safetensors_path = if model_path.is_dir() {
        // Find safetensors file
        let mut path = None;
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            if entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
            {
                path = Some(entry.path());
                break;
            }
        }
        path.context("No .safetensors file found in model directory")?
    } else {
        model_path.to_path_buf()
    };

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::BF16, &device)? };

    let vae = VaeDecoder::new(vb.pp("vae.decoder"))?;

    // Decode (no timestep conditioning for now)
    info!("Decoding latents...");
    let decode_timestep: Option<f64> = Some(0.05); // typical value

    let video = vae.decode(&latents, decode_timestep)?;
    info!("Decoded video shape: {:?}", video.dims());

    // Save frames
    let output_dir = Path::new(&args.output);
    std::fs::create_dir_all(output_dir)?;

    let num_frames = video.dim(2)?;
    info!("Saving {} frames...", num_frames);

    // Save first and last frames
    save_frame(&video, 0, output_dir)?;
    if num_frames > 1 {
        save_frame(&video, num_frames - 1, output_dir)?;
    }

    info!("Done! Check output in {:?}", output_dir);

    Ok(())
}
