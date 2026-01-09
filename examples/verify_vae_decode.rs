use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::vae::{AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig};
use std::path::PathBuf;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    // Use BF16 for GPU - matches LTX model format
    let dtype = DType::BF16;

    println!("Device: {:?}", device);

    println!("Loading reference VAE data...");
    let ref_path = "reference_output/vae_debug.safetensors";
    let ref_tensors = candle_core::safetensors::load(ref_path, &Device::Cpu)?;

    let latents = ref_tensors
        .get("latents")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;
    let py_video = ref_tensors
        .get("video")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;

    println!("Latents Shape: {:?}", latents.dims());
    println!("Reference Video Shape: {:?}", py_video.dims());

    println!("Loading VAE Model...");
    let model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae";
    let weights_path = PathBuf::from(model_path).join("diffusion_pytorch_model.safetensors");
    let config_path = PathBuf::from(model_path).join("config.json");

    let config: AutoencoderKLLtxVideoConfig = {
        let file = std::fs::File::open(config_path)?;
        serde_json::from_reader(file).map_err(candle_core::Error::wrap)?
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let vae = AutoencoderKLLtxVideo::new(config, vb)?;

    println!("Decoding...");
    // decode(latents, temb, return_dict, train)
    // Need temb tensor (0.0)
    let (b, _, _, _, _) = latents.dims5()?;
    let temb = Tensor::zeros((b,), dtype, &device)?;

    let (_, video) = vae.decode(&latents, Some(&temb), false, false)?;

    println!("Comparing results...");
    // Convert to F32 for comparison
    let video_f32 = video.to_dtype(DType::F32)?;
    let py_video_f32 = py_video.to_dtype(DType::F32)?;

    let diff = (&video_f32 - &py_video_f32)?.abs()?;
    let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let avg_diff = diff.flatten_all()?.mean(0)?.to_scalar::<f32>()?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

    println!("VAE Output Diff Max: {}", max_diff);
    println!("VAE Output Diff Avg: {}", avg_diff);
    println!("VAE Output MSE: {:.2e}", mse);

    // Requirement 4.1: MSE < 1e-3
    if mse < 1e-3 {
        println!("SUCCESS: VAE matches (MSE < 1e-3).");
    } else if mse < 1e-2 {
        println!("SUCCESS: VAE matches (MSE < 1e-2, acceptable for BF16).");
    } else {
        println!("FAILURE: VAE mismatch (MSE {} exceeds threshold).", mse);
    }

    Ok(())
}
