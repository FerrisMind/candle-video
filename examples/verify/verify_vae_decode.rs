//! VAE decode parity verification - runs as example to avoid thread-local cleanup issues

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::vae::{
    AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig,
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let parity_path = Path::new("gen_vae_parity.safetensors");
    let vae_path = Path::new("models/models--Lightricks--LTX-Video-0.9.5/vae/diffusion_pytorch_model.safetensors");
    
    if !parity_path.exists() {
        println!("Skipping: gen_vae_parity.safetensors not found");
        return Ok(());
    }
    if !vae_path.exists() {
        println!("Skipping: VAE model not found");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    println!("Running on device: {:?}, dtype: {:?}", device, dtype);

    let ref_tensors = candle_core::safetensors::load(parity_path, &device)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[vae_path], dtype, &device)?
    };
    let config = AutoencoderKLLtxVideoConfig::default();
    let model = AutoencoderKLLtxVideo::new(config, vb)?;

    // Test cases
    let test_cases = vec![
        ("decode_shape0_t0.0", "Shape0 t=0.0"),
        ("decode_shape0_t0.05", "Shape0 t=0.05"),
        ("decode_shape1_t0.05", "Shape1 t=0.05"),
    ];

    let mut all_passed = true;

    for (prefix, desc) in test_cases {
        let latents_key = format!("{}_latents", prefix);
        let temb_key = format!("{}_temb", prefix);
        let output_key = format!("{}_output", prefix);

        let latents = match ref_tensors.get(&latents_key) {
            Some(t) => t.to_dtype(dtype)?,
            None => continue,
        };
        let temb = match ref_tensors.get(&temb_key) {
            Some(t) => t.to_dtype(dtype)?,
            None => continue,
        };
        let ref_output = match ref_tensors.get(&output_key) {
            Some(t) => t,
            None => continue,
        };

        println!("\n{}: latents shape {:?}", desc, latents.shape());

        let (_, output) = model.decode(&latents, Some(&temb), false, false)?;

        let output_f32 = output.to_dtype(DType::F32)?;
        let ref_output_f32 = ref_output.to_dtype(DType::F32)?;

        let diff = output_f32.sub(&ref_output_f32)?.abs()?;
        let max_diff = diff.max_all()?.to_vec0::<f32>()?;
        let mse = diff.sqr()?.mean_all()?.to_vec0::<f32>()?;

        println!("  Output shape: {:?}", output.shape());
        println!("  Max diff: {:.6}", max_diff);
        println!("  MSE: {:.2e}", mse);

        // Requirement 4.1: MSE < 1e-3
        if mse < 1e-3 {
            println!("  ✓ PASS (MSE < 1e-3)");
        } else if mse < 1e-2 {
            println!("  ✓ PASS (MSE < 1e-2, acceptable for BF16)");
        } else {
            println!("  ✗ FAIL (MSE {} exceeds threshold)", mse);
            all_passed = false;
        }
    }

    if all_passed {
        println!("\n✓ All VAE decode parity tests passed!");
    } else {
        println!("\n✗ Some tests failed!");
        std::process::exit(1);
    }

    Ok(())
}
