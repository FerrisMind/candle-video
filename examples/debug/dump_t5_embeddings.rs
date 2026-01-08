use candle_core::{Device, Tensor};
use candle_video::models::ltx_video::quantized_t5_encoder::QuantizedT5EncoderModel;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Running on device: {:?}", device);

    let model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5";
    let t5_path = format!(
        "{}/text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf",
        model_path
    );
    let tokenizer_path = format!("{}/text_encoder_gguf/tokenizer.json", model_path);

    println!("Loading T5 GGUF from {}", t5_path);
    let t5_model = QuantizedT5EncoderModel::load(&t5_path, &device)?;

    println!("Loading tokenizer from {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    let prompt = "The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.";
    let negative_prompt = "worst quality, bad quality, deformation, low resolution";

    println!("Encoding prompts...");

    // Encode prompt
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut mask: Vec<u32> = encoding.get_attention_mask().to_vec();

    // Pad/truncate to 128
    let max_len = 128;
    ids.truncate(max_len);
    mask.truncate(max_len);
    ids.resize(max_len, 0);
    mask.resize(max_len, 0);

    let input_ids = Tensor::new(ids.clone(), &device)?.reshape((1, max_len))?;
    let attention_mask = Tensor::new(mask.clone(), &device)?.reshape((1, max_len))?;

    let prompt_embeds = t5_model.forward(&input_ids, Some(&attention_mask))?;

    // Encode negative prompt
    let neg_encoding = tokenizer
        .encode(negative_prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut neg_ids: Vec<u32> = neg_encoding.get_ids().to_vec();
    let mut neg_mask: Vec<u32> = neg_encoding.get_attention_mask().to_vec();
    neg_ids.truncate(max_len);
    neg_mask.truncate(max_len);
    neg_ids.resize(max_len, 0);
    neg_mask.resize(max_len, 0);

    let neg_input_ids = Tensor::new(neg_ids, &device)?.reshape((1, max_len))?;
    let neg_attention_mask = Tensor::new(neg_mask, &device)?.reshape((1, max_len))?;

    let negative_prompt_embeds = t5_model.forward(&neg_input_ids, Some(&neg_attention_mask))?;

    // Save mask as raw 1.0/0.0 values for the pipeline
    let prompt_attention_mask = attention_mask.to_dtype(candle_core::DType::F32)?;
    let negative_prompt_attention_mask = neg_attention_mask.to_dtype(candle_core::DType::F32)?;

    println!("\nPrompt embeddings shape: {:?}", prompt_embeds.dims());
    if let Ok(vals) = prompt_embeds.flatten_all()?.to_vec1::<f32>() {
        println!("Prompt embeddings first 10: {:?}", &vals[..10]);
        println!(
            "Prompt embeddings mean: {:.6}",
            vals.iter().sum::<f32>() / vals.len() as f32
        );
    }

    // Save to safetensors
    println!("\nSaving embeddings to t5_gguf_embeddings.safetensors...");

    let prompt_embeds_cpu = prompt_embeds.to_device(&Device::Cpu)?;
    let negative_prompt_embeds_cpu = negative_prompt_embeds.to_device(&Device::Cpu)?;
    let prompt_attention_mask_cpu = prompt_attention_mask.to_device(&Device::Cpu)?;
    let negative_prompt_attention_mask_cpu =
        negative_prompt_attention_mask.to_device(&Device::Cpu)?;

    // Convert to safetensors format
    use safetensors::serialize;
    use std::collections::HashMap;

    let pe_data = prompt_embeds_cpu.flatten_all()?.to_vec1::<f32>()?;
    let ne_data = negative_prompt_embeds_cpu.flatten_all()?.to_vec1::<f32>()?;
    let pm_data = prompt_attention_mask_cpu.flatten_all()?.to_vec1::<f32>()?;
    let nm_data = negative_prompt_attention_mask_cpu
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();

    // Write manually with proper shapes
    let pe_shape: Vec<usize> = prompt_embeds_cpu.dims().to_vec();
    let ne_shape: Vec<usize> = negative_prompt_embeds_cpu.dims().to_vec();
    let pm_shape: Vec<usize> = prompt_attention_mask_cpu.dims().to_vec();
    let nm_shape: Vec<usize> = negative_prompt_attention_mask_cpu.dims().to_vec();

    let pe_bytes: Vec<u8> = pe_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let ne_bytes: Vec<u8> = ne_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let pm_bytes: Vec<u8> = pm_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let nm_bytes: Vec<u8> = nm_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    tensors.insert(
        "prompt_embeds".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, pe_shape, &pe_bytes)?,
    );
    tensors.insert(
        "negative_prompt_embeds".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, ne_shape, &ne_bytes)?,
    );
    tensors.insert(
        "prompt_attention_mask".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, pm_shape, &pm_bytes)?,
    );
    tensors.insert(
        "negative_prompt_attention_mask".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, nm_shape, &nm_bytes)?,
    );

    let serialized = serialize(&tensors, None)?;
    std::fs::write("t5_gguf_embeddings.safetensors", serialized)?;

    println!("Done! Saved to t5_gguf_embeddings.safetensors");

    Ok(())
}
