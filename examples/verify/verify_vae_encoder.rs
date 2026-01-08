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
    println!("Loading verification data from vae_encode_verification.safetensors...");
    let verification_data =
        candle_core::safetensors::load("vae_encode_verification.safetensors", &device)?;
    let video_input = verification_data
        .get("video_input")
        .expect("Missing video_input")
        .clone();
    let latents_mean_ref = verification_data
        .get("latents_mean")
        .expect("Missing latents_mean")
        .clone();

    println!("Video Input shape: {:?}", video_input.dims());
    println!(
        "Latents Mean Reference shape: {:?}",
        latents_mean_ref.dims()
    );

    // Forward Pass (Encode)
    println!("Running VAE Encode...");
    // encode returns (Option<AutoencoderKLOutput>, DiagonalGaussianDistribution)
    // we want the distribution
    let (_, posterior) = vae.encode(&video_input, false, false)?;

    // Get mean from distribution
    let latents_mean_rust = posterior.mean;

    println!("Rust Latents Mean shape: {:?}", latents_mean_rust.dims());

    // Compare
    let diff = (&latents_mean_rust - &latents_mean_ref)?;
    let abs_diff = diff.abs()?;
    let max_diff = abs_diff
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mean_diff = abs_diff.mean_all()?.to_scalar::<f32>()?;

    println!("\n--- COMPARISON RESULTS ---");
    println!("Max Absolute Difference: {:.6}", max_diff);
    println!("Mean Absolute Difference: {:.6}", mean_diff);

    if mean_diff < 0.05 {
        // Encoder should be quite precise
        println!("\nSUCCESS: Rust VAE Encoder matches Python VAE within tolerance!");
    } else {
        println!("\nFAILURE: Significant difference detected.");
        // Print stats
        println!(
            "Rust Mean: {:.4}",
            latents_mean_rust.mean_all()?.to_scalar::<f32>()?
        );
        println!(
            "Py Mean: {:.4}",
            latents_mean_ref.mean_all()?.to_scalar::<f32>()?
        );
    }

    Ok(())
}
