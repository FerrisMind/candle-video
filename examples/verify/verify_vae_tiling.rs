use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::vae::{AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig};
use std::path::PathBuf;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = DType::F32;

    println!("Device: {:?}", device);

    println!("Loading reference VAE tiling data...");
    let ref_path = "reference_output/vae_tiling.safetensors";
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

    println!("Loading VAE Model...");
    let model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae";
    let weights_path = PathBuf::from(model_path).join("diffusion_pytorch_model.safetensors");
    let config_path = PathBuf::from(model_path).join("config.json");

    let config: AutoencoderKLLtxVideoConfig = {
        let file = std::fs::File::open(config_path)?;
        serde_json::from_reader(file).map_err(candle_core::Error::wrap)?
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let mut vae = AutoencoderKLLtxVideo::new(config, vb)?;

    // Enable tiling manually (Rust struct fields)
    // tile_sample_min_height = 128
    // tile_sample_min_width = 128
    // tile_sample_min_num_frames = 8? Script said 16 but latents produce 9?
    // Script: tile_sample_min_num_frames=8 (I set in comment but 16 in code? No, let's check script)
    // Script: `tile_sample_min_num_frames=16`. Output is 9 frames.
    // 9 < 16, so TEMPORAL tiling NOT triggered.
    // Only SPATIAL tiling.

    vae.enable_tiling(
        Some(128), // min_h
        Some(128), // min_w
        Some(16),  // min_f
        Some(96),  // stride_h
        Some(96),  // stride_w
        Some(4),   // stride_f
    );

    println!("Decoding with tiling...");
    let (b, _, _, _, _) = latents.dims5()?;
    let temb = Tensor::zeros((b,), dtype, &device)?;

    let (_, video) = vae.decode(&latents, Some(&temb), false, false)?;

    println!("Comparing results...");
    let diff = (&video - &py_video)?.abs()?;
    let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let avg_diff = diff.flatten_all()?.mean(0)?.to_scalar::<f32>()?;

    println!("VAE Tiling Diff Max: {}", max_diff);
    println!("VAE Tiling Diff Avg: {}", avg_diff);

    if max_diff < 1e-3 {
        println!("SUCCESS: VAE Tiling matches.");
    } else {
        println!("FAILURE: VAE Tiling mismatch.");
    }

    Ok(())
}
