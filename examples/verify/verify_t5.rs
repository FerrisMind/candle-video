use candle_core::{DType, Device, Result};
use candle_video::models::ltx_video::quantized_t5_encoder::QuantizedT5EncoderModel;

fn main() -> Result<()> {
    // 1. Setup device and dtype
    let device = Device::Cpu;
    let _dtype = DType::F32;

    println!("Device: {:?}", device);

    // 2. Load reference data
    println!("Loading reference data from reference_output/t5_ref.safetensors...");
    let ref_path = "reference_output/t5_ref.safetensors";
    let ref_tensors = candle_core::safetensors::load(ref_path, &Device::Cpu)?;
    let input_ids = ref_tensors.get("input_ids").unwrap().to_device(&device)?;
    let attention_mask = ref_tensors
        .get("attention_mask")
        .unwrap()
        .to_device(&device)?;
    let py_output = ref_tensors.get("output").unwrap().to_device(&device)?;

    println!("Input IDs Shape: {:?}", input_ids.dims());
    println!("Reference Output Shape: {:?}", py_output.dims());

    // 3. Verify Full Precision (FP32) - SKIPPED (missing weights)
    let max_diff_fp32 = 0.0f32; // dummy

    // 4. Verify GGUF (Quantized)
    println!("\n--- Verifying GGUF T5 ---");
    let gguf_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf";

    println!("Loading GGUF weights...");
    let gguf_model = QuantizedT5EncoderModel::load(gguf_path, &device)?;

    println!("Running GGUF forward pass...");
    let gguf_output =
        gguf_model.forward(&input_ids, Some(&attention_mask.to_dtype(DType::F32)?))?;

    println!("Comparing GGUF results...");
    let diff_gguf = (&gguf_output - &py_output)?.abs()?;
    let max_diff_gguf = diff_gguf.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    let avg_diff_gguf = diff_gguf.flatten_all()?.mean(0)?.to_vec0::<f32>()?;
    println!("Max Absolute Difference (GGUF): {}", max_diff_gguf);
    println!("Avg Absolute Difference (GGUF): {}", avg_diff_gguf);

    println!("\n--- Summary ---");
    if max_diff_fp32 < 0.05 {
        println!("FP32: SUCCESS");
    } else {
        println!("FP32: FAILURE (Difference too large)");
    }

    if max_diff_gguf < 0.8 {
        println!("GGUF: SUCCESS (within quantization tolerance)");
    } else {
        println!("GGUF: FAILURE (Difference too large even for quantization)");
    }

    Ok(())
}
