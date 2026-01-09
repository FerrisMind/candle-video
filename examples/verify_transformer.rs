use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::ltx_transformer::{
    LtxVideoTransformer3DModel, LtxVideoTransformer3DModelConfig,
};
use std::path::Path;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = DType::F32;

    // 1. Load weights
    let model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/transformer";
    let weights_path = Path::new(model_path).join("diffusion_pytorch_model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };

    // 2. Load config
    let config_path = Path::new(model_path).join("config.json");
    let config: LtxVideoTransformer3DModelConfig = {
        let file = std::fs::File::open(config_path)?;
        serde_json::from_reader(file).map_err(candle_core::Error::wrap)?
    };

    println!("Initializing model...");
    let model = LtxVideoTransformer3DModel::new(&config, vb)?;

    // 3. Load reference data
    println!("Loading reference data from dit_ref.safetensors...");
    let ref_path = "dit_ref.safetensors";
    if !Path::new(ref_path).exists() {
        println!("Error: {} not found. Run python script first.", ref_path);
        return Ok(());
    }
    let ref_tensors = candle_core::safetensors::load(ref_path, &device)?;

    let hidden_states = ref_tensors.get("hidden_states").unwrap().to_dtype(dtype)?;
    let encoder_hidden_states = ref_tensors
        .get("encoder_hidden_states")
        .unwrap()
        .to_dtype(dtype)?;
    let timestep = ref_tensors.get("timestep").unwrap().to_dtype(dtype)?;
    let encoder_attention_mask = ref_tensors
        .get("encoder_attention_mask")
        .unwrap()
        .to_dtype(dtype)?;
    let py_output = ref_tensors.get("output").unwrap().to_dtype(dtype)?;

    let num_frames = ref_tensors.get("num_frames").unwrap().to_vec1::<i64>()?[0] as usize;
    let height = ref_tensors.get("height").unwrap().to_vec1::<i64>()?[0] as usize;
    let width = ref_tensors.get("width").unwrap().to_vec1::<i64>()?[0] as usize;

    let rope_interpolation_scale = Some((20.0 / 25.0, 32.0, 32.0));

    // 4. Run forward
    println!("Running Rust forward pass...");
    let rust_output = model.forward(
        &hidden_states,
        &encoder_hidden_states,
        &timestep,
        Some(&encoder_attention_mask),
        num_frames,
        height,
        width,
        rope_interpolation_scale,
        None,
        None, // skip_layer_mask
    )?;

    // 5. Compare
    println!("Comparing results...");
    let diff = (&rust_output - &py_output)?.abs()?;
    let max_diff = diff.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    let avg_diff = diff.flatten_all()?.mean(0)?.to_vec0::<f32>()?;

    println!("Max absolute difference: {}", max_diff);
    println!("Avg absolute difference: {}", avg_diff);

    if max_diff < 0.1 {
        println!("SUCCESS: Difference is within tolerance (0.1)");
    } else {
        println!("FAILURE: Difference is too large!");
    }

    Ok(())
}
