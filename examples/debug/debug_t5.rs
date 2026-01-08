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

    println!("Tokenizing prompt...");

    // Encode prompt
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();

    // Pad/truncate to 128
    let max_len = 128;
    ids.truncate(max_len);
    ids.resize(max_len, 0);

    println!("Input IDs first 20: {:?}", &ids[..20]);

    let input_ids = Tensor::new(ids.clone(), &device)?.reshape((1, max_len))?;

    // Get embedding weights and do lookup manually to debug
    println!("\n=== Debug: Token Embeddings ===");

    // Access embedding from model (need to add a method for this)
    // For now, forward and print intermediate values
    let embeddings = t5_model.forward(&input_ids, None)?;

    if let Ok(vals) = embeddings.flatten_all()?.to_vec1::<f32>() {
        println!("Final output first 10: {:?}", &vals[..10]);
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        let variance: f32 =
            vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        let std = variance.sqrt();
        println!("Final output mean: {:.6}, std: {:.6}", mean, std);
    }

    // Compare with Python
    println!("\n=== Python Reference (from debug_t5_layers.py) ===");
    println!(
        "Token embeddings first 10: [1.617, 4.469, -3.484, 1.070, 2.031, 0.660, 4.531, -0.676, 0.124, -1.453]"
    );
    println!("Token embeddings mean: 0.007411, std: 5.451715");

    Ok(())
}
