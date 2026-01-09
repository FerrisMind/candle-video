use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::ltx_transformer::{
    LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig,
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    local_weights: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Use CPU for verification to ensure maximum precision and match Python reference exactly
    let device = Device::Cpu;

    println!("Loading DiT weights from: {}", args.local_weights);
    let weights_path =
        PathBuf::from(&args.local_weights).join("transformer/diffusion_pytorch_model.safetensors");
    let _config_path = PathBuf::from(&args.local_weights).join("transformer/config.json");

    // Default config matching Python test
    let config = LtxVideoTransformer3DModelConfig::default();

    // Load Weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

    // Init DiT
    let dit = LtxVideoTransformer3DModel::new(&config, vb)?;

    println!("DiT loaded successfully.");

    // Load Verification Data
    println!("Loading verification data from dit_verification.safetensors...");
    let verification_data =
        candle_core::safetensors::load("dit_verification.safetensors", &device)?;
    let hidden_states = verification_data
        .get("hidden_states")
        .expect("Missing hidden_states")
        .clone();
    let encoder_hidden_states = verification_data
        .get("encoder_hidden_states")
        .expect("Missing encoder_hidden_states")
        .clone();
    let timestep = verification_data
        .get("timestep")
        .expect("Missing timestep")
        .clone();
    let encoder_attention_mask = verification_data
        .get("encoder_attention_mask")
        .expect("Missing encoder_attention_mask")
        .clone();
    let py_output = verification_data
        .get("output")
        .expect("Missing output")
        .clone();

    println!("Inputs:");
    println!("  hidden_states: {:?}", hidden_states.dims());
    println!(
        "  encoder_hidden_states: {:?}",
        encoder_hidden_states.dims()
    );
    println!("  timestep: {:?}", timestep.dims());
    println!(
        "  encoder_attention_mask: {:?}",
        encoder_attention_mask.dims()
    );

    // Run Rust DiT
    println!("\nRunning Rust DiT forward pass...");
    // num_frames=2, height=8, width=8 as in python script
    let rust_output = dit.forward(
        &hidden_states,
        &encoder_hidden_states,
        &timestep,
        Some(&encoder_attention_mask),
        2,
        8,
        8,
        Some((1.0, 1.0, 1.0)),
        None,
        None, // skip_layer_mask
    )?;

    println!("Rust Output shape: {:?}", rust_output.dims());
    println!(
        "Rust Output mean: {:.6}",
        rust_output.mean_all()?.to_scalar::<f32>()?
    );
    println!(
        "Py Output mean:   {:.6}",
        py_output.mean_all()?.to_scalar::<f32>()?
    );

    // Comparison
    let diff = (&rust_output - &py_output)?;
    let mad = diff.abs()?.mean_all()?.to_scalar::<f32>()?;
    let max_diff = diff
        .abs()?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .cloned()
        .fold(0f32, f32::max);

    println!("\n--- COMPARISON RESULTS ---");
    println!("Max Absolute Difference: {:.6}", max_diff);
    println!("Mean Absolute Difference: {:.6}", mad);

    // Value inspection
    let rust_vals = rust_output.flatten_all()?.to_vec1::<f32>()?;
    let py_vals = py_output.flatten_all()?.to_vec1::<f32>()?;
    println!("\nFirst 5 Values Comparison:");
    for i in 0..5 {
        println!(
            "  Idx {}: Rust={:.6}, Py={:.6}, Diff={:.6}",
            i,
            rust_vals[i],
            py_vals[i],
            (rust_vals[i] - py_vals[i]).abs()
        );
    }

    if mad < 1e-4 {
        println!("SUCCESS: Rust DiT matches Python within tolerance!");
    } else {
        println!("FAILURE: Significant mismatch detected in DiT output.");
    }

    Ok(())
}
