use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::vae::AutoencoderKLLtxVideoConfig;
use clap::Parser;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    local_weights: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0)?;

    println!("Loading VAE weights from: {}", args.local_weights);
    let weights_path =
        PathBuf::from(&args.local_weights).join("vae/diffusion_pytorch_model.safetensors");
    let config_path = PathBuf::from(&args.local_weights).join("vae/config.json");

    // Load Config
    let config_file =
        File::open(&config_path).map_err(|e| anyhow::anyhow!("Failed to open config: {}", e))?;
    let config: AutoencoderKLLtxVideoConfig = serde_json::from_reader(config_file)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    // Load Weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

    // Init VAE
    let vae = candle_video::models::ltx_video::vae::AutoencoderKLLtxVideo::new(config, vb)?;

    println!("VAE loaded successfully.");

    // Load Verification Data
    println!("Loading verification data from vae_roundtrip_verification.safetensors...");
    let verification_data =
        candle_core::safetensors::load("vae_roundtrip_verification.safetensors", &device)?;
    let video_input = verification_data
        .get("video_input")
        .expect("Missing video_input")
        .clone();
    let py_latents = verification_data
        .get("latents")
        .expect("Missing latents")
        .clone();
    let py_output = verification_data
        .get("video_output")
        .expect("Missing video_output")
        .clone();
    let temb = verification_data.get("temb").cloned();

    println!("Video Input shape: {:?}", video_input.dims());
    println!("Python Latents shape: {:?}", py_latents.dims());
    println!("Python Output shape: {:?}", py_output.dims());

    // 1. Rust Encode
    println!("\n--- STEP 1: RUST ENCODE ---");
    let (_, posterior) = vae.encode(&video_input, false, false)?;
    let rust_latents = posterior.mean; // match posterior.mode() in python
    println!("Rust Latents shape: {:?}", rust_latents.dims());

    let diff_latents = (&rust_latents - &py_latents)?;
    let mad_latents = diff_latents.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("Latents Mean Absolute Difference: {:.6}", mad_latents);

    // 2. Rust Decode
    println!("\n--- STEP 2: RUST DECODE ---");
    // We decode our OWN latents to see the full pipeline drift
    let (_, rust_output) = vae.decode(&rust_latents, temb.as_ref(), false, false)?;
    println!("Rust Output shape: {:?}", rust_output.dims());

    let diff_output = (&rust_output - &py_output)?;
    let mad_output = diff_output.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("Output Mean Absolute Difference: {:.6}", mad_output);

    // Comparison Summary
    println!("\n--- FINAL SUMMARY ---");
    if mad_latents < 0.05 && mad_output < 0.1 {
        println!("SUCCESS: Rust VAE Full Pipeline matches Python within tolerances!");
    } else {
        println!("FAILURE: Significant drift detected in the full pipeline.");
        println!(
            "Rust Output Mean: {:.4}",
            rust_output.mean_all()?.to_scalar::<f32>()?
        );
        println!(
            "Py Output Mean: {:.4}",
            py_output.mean_all()?.to_scalar::<f32>()?
        );
    }

    Ok(())
}
