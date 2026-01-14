// Minimal test for Wan transformer forward pass
//
// This tests the transformer with small dimensions to verify it works
// before trying larger resolutions.
//
// Usage:
//   cargo run --example wan-test-transformer --release --features flash-attn,cudnn

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{WanTransformer3DConfig, load_transformer};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("Wan Transformer Test");
    println!("====================");

    // Check for model
    let model_path = Path::new("models/Wan2.1-T2V-1.3B");
    if !model_path.exists() {
        println!("Model not found at {:?}", model_path);
        println!("Please download the model first.");
        return Ok(());
    }

    let transformer_path = model_path.join("wan2.1_t2v_1.3B_fp16.safetensors");
    if !transformer_path.exists() {
        println!("Transformer weights not found at {:?}", transformer_path);
        return Ok(());
    }

    // Setup device
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    println!("Device: CUDA");
    println!("Dtype: BF16");

    // Load transformer
    println!("\nLoading transformer...");
    let config = WanTransformer3DConfig::wan_t2v_1_3b();
    let transformer = load_transformer(&transformer_path, config, &device, dtype)?;
    println!("✓ Transformer loaded");

    // Test with small dimensions
    // 17 frames -> 5 latent frames
    // 256x256 -> 32x32 latent -> 16x16 patches
    // Sequence length: 5 * 16 * 16 = 1280 tokens
    let num_frames = 17;
    let height = 256;
    let width = 256;

    let latent_frames = (num_frames - 1) / 4 + 1; // 5
    let latent_h = height / 8; // 32
    let latent_w = width / 8; // 32

    let seq_len = latent_frames * (latent_h / 2) * (latent_w / 2); // 5 * 16 * 16 = 1280

    println!("\nTest configuration:");
    println!("  Video: {} frames × {}×{}", num_frames, height, width);
    println!("  Latent: {} × {} × {}", latent_frames, latent_h, latent_w);
    println!("  Sequence length: {} tokens", seq_len);

    // Create test inputs
    println!("\nCreating test inputs...");

    let hidden_states = Tensor::randn(
        0f32,
        1f32,
        (1, 16, latent_frames, latent_h, latent_w),
        &device,
    )?
    .to_dtype(dtype)?;
    println!("  hidden_states: {:?}", hidden_states.dims());

    let timestep = Tensor::from_vec(vec![500f32], (1,), &device)?.to_dtype(dtype)?;
    println!("  timestep: {:?}", timestep.dims());

    let encoder_hidden_states =
        Tensor::randn(0f32, 1f32, (1, 512, 4096), &device)?.to_dtype(dtype)?;
    println!(
        "  encoder_hidden_states: {:?}",
        encoder_hidden_states.dims()
    );

    // Run forward pass
    println!("\nRunning forward pass...");

    let output = transformer.forward(
        &hidden_states,
        &timestep,
        &encoder_hidden_states,
        None,
        false,
    )?;

    let output_tensor = match output {
        Ok(out) => out.sample,
        Err(tensor) => tensor,
    };

    println!("✓ Forward pass complete");
    println!("  Output shape: {:?}", output_tensor.dims());

    // Verify output shape matches input
    let expected_shape = hidden_states.dims();
    let actual_shape = output_tensor.dims();

    if expected_shape == actual_shape {
        println!("✓ Output shape matches input shape");
    } else {
        println!(
            "✗ Shape mismatch: expected {:?}, got {:?}",
            expected_shape, actual_shape
        );
    }

    println!("\nTest complete!");

    Ok(())
}
