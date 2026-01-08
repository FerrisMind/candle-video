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
    println!("Loading verification data from vae_verification.safetensors...");
    let verification_data =
        candle_core::safetensors::load("vae_verification.safetensors", &device)?;
    let latents = verification_data
        .get("latents")
        .expect("Missing latents")
        .clone();
    let python_decoded = verification_data
        .get("decoded")
        .expect("Missing decoded")
        .clone();
    // temb is scalar in python [1], but here we might need adjustment.
    // Wait, python script saved temb as tensor([0.05]).
    let temb = verification_data.get("temb").cloned();

    println!("Latents shape: {:?}", latents.dims());
    println!("Reference Output shape: {:?}", python_decoded.dims());
    if let Some(ref t) = temb {
        println!(
            "temb shape: {:?}, value: {:?}",
            t.dims(),
            t.to_vec1::<f32>()
        );
    } else {
        println!("temb: None");
    }

    // Forward Pass
    println!("Running VAE Decode...");
    let (_, rust_decoded) = vae.decode(&latents, temb.as_ref(), false, false)?;

    println!("Rust Output shape: {:?}", rust_decoded.dims());

    // Compare
    let diff = (&rust_decoded - &python_decoded)?;
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

    if max_diff < 0.5 {
        // Tolerance for 3D VAE - float precision can accumulate
        println!("\nSUCCESS: Rust VAE matches Python VAE within tolerance!");
        println!(
            "Rust Mean: {:.4}",
            rust_decoded.mean_all()?.to_scalar::<f32>()?
        );
        println!(
            "Py Mean: {:.4}",
            python_decoded.mean_all()?.to_scalar::<f32>()?
        );
    } else {
        println!("\nFAILURE: Significant difference detected.");
        // Print stats
        println!(
            "Rust Mean: {:.4}",
            rust_decoded.mean_all()?.to_scalar::<f32>()?
        );
        println!(
            "Py Mean: {:.4}",
            python_decoded.mean_all()?.to_scalar::<f32>()?
        );
    }

    Ok(())
}
