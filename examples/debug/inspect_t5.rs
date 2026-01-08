use candle_core::{Device, IndexOp, Tensor};
use candle_video::models::ltx_video::quantized_t5_encoder::QuantizedT5EncoderModel;
use candle_video::models::ltx_video::t2v_pipeline::Tokenizer;
use clap::Parser;
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_path: String,

    #[arg(long)]
    tokenizer_path: String,

    #[arg(long, default_value = "A red cube")]
    prompt: String,
}

struct TokenizerAdapter {
    tokenizer: HfTokenizer,
    device: Device,
}

impl Tokenizer for TokenizerAdapter {
    fn encode_batch(
        &self,
        prompts: &[String],
        max_length: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let encodings = self
            .tokenizer
            .encode_batch(prompts.to_vec(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut ids_vec = Vec::new();
        let mut mask_vec = Vec::new();
        for e in encodings {
            let mut ids = e.get_ids().to_vec();
            let mut mask = e.get_attention_mask().to_vec();
            if ids.len() > max_length {
                ids.truncate(max_length);
                mask.truncate(max_length);
            } else {
                while ids.len() < max_length {
                    ids.push(0);
                    mask.push(0);
                }
            }
            ids_vec.push(Tensor::new(ids, &self.device)?);
            mask_vec.push(Tensor::new(mask, &self.device)?);
        }
        let ids = Tensor::stack(&ids_vec, 0)?;
        let mask = Tensor::stack(&mask_vec, 0)?;
        Ok((ids, mask))
    }

    fn model_max_length(&self) -> usize {
        128
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("Loading T5 from {}", args.model_path);
    let t5 = QuantizedT5EncoderModel::load(std::path::Path::new(&args.model_path), &device)?;

    println!("Loading Tokenizer from {}", args.tokenizer_path);
    let tokenizer = HfTokenizer::from_file(&args.tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
    let adapter = TokenizerAdapter {
        tokenizer,
        device: device.clone(),
    };

    println!("Encoding prompt: '{}'", args.prompt);
    let (ids, mask) = adapter.encode_batch(std::slice::from_ref(&args.prompt), 128)?;

    println!(
        "Token IDs (first 20): {:?}",
        ids.i((0, 0..20))?.to_vec1::<u32>()?
    );

    let embeddings = t5.forward(&ids, Some(&mask))?;

    println!("Embeddings shape: {:?}", embeddings.shape());

    let first_token_emb = embeddings.i((0, 0, ..))?;

    println!(
        "First token embedding mean: {}",
        first_token_emb.mean_all()?
    );
    println!(
        "First token embedding std: {}",
        (first_token_emb.sqr()?.mean_all()? - first_token_emb.mean_all()?.sqr()?)?.sqrt()?
    );

    // Save to safetensors for precise comparison
    let save_path = "t5_embeddings_rust.safetensors";
    println!("Saving embeddings to {}", save_path);
    candle_core::safetensors::save(
        &std::collections::HashMap::from([("prompt_embeds".to_string(), embeddings)]),
        save_path,
    )?;

    Ok(())
}
