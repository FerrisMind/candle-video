//! Export text embeddings from candle T5 encoder for diffusers comparison
//!
//! Usage: cargo run --bin export-embeddings

use anyhow::Result;
use candle_core::{DType, Device};
use candle_video::QuantizedT5Encoder;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("=== Exporting Text Embeddings ===");

    let model_dir = PathBuf::from("ltxv-2b-0.9.8-distilled/T5-XXL-8bit");
    let gguf_path = PathBuf::from("ltxv-2b-0.9.8-distilled/t5-v1_1-xxl-encoder-Q5_K_M.gguf");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");

    let prompt = "A red apple on a wooden table";
    let negative_prompt = "low quality";
    let max_seq_len = 128;

    println!("Model: {:?}", gguf_path);
    println!("Tokenizer: {:?}", tokenizer_path);
    println!("Config: {:?}", config_path);
    println!("Prompt: {}", prompt);
    println!("Negative: {}", negative_prompt);

    let device = Device::Cpu; // Use CPU to match diffusers exactly

    // Load T5 encoder
    println!("\nLoading T5 encoder...");
    let mut encoder = QuantizedT5Encoder::load(
        &gguf_path,
        &tokenizer_path,
        &config_path,
        &device,
        max_seq_len,
    )?;

    // Encode prompts
    println!("Encoding prompts...");
    let neg_embeds = encoder.encode(negative_prompt)?;
    let pos_embeds = encoder.encode(prompt)?;

    println!("Negative embeds: {:?}", neg_embeds.shape());
    println!("Positive embeds: {:?}", pos_embeds.shape());

    // Concatenate [negative, positive] for CFG
    let embeds = candle_core::Tensor::cat(&[&neg_embeds, &pos_embeds], 0)?;

    println!("Combined embeds: {:?}", embeds.shape());

    // Create attention mask (all ones for now - full attention)
    let (batch, seq, _hidden) = embeds.dims3()?;
    let mask = candle_core::Tensor::ones((batch, seq), DType::I64, &device)?;

    println!("Mask: {:?}", mask.shape());

    // Convert to f32 for saving
    let embeds_f32 = embeds.to_dtype(DType::F32)?;
    let embeds_vec: Vec<f32> = embeds_f32.flatten_all()?.to_vec1()?;

    let mask_vec: Vec<i64> = mask.flatten_all()?.to_vec1()?;

    // Save embeddings
    let out_dir = PathBuf::from("output/diffusers_ref");
    std::fs::create_dir_all(&out_dir)?;

    // Save as raw binary format
    let embeds_path = out_dir.join("prompt_embeds.bin");
    let mask_path = out_dir.join("prompt_mask.bin");

    // Write raw f32 bytes
    let mut file = File::create(&embeds_path)?;
    for v in &embeds_vec {
        file.write_all(&v.to_le_bytes())?;
    }
    println!(
        "Saved embeddings to {:?} ({} floats)",
        embeds_path,
        embeds_vec.len()
    );

    // Write raw i64 bytes
    let mut file = File::create(&mask_path)?;
    for v in &mask_vec {
        file.write_all(&v.to_le_bytes())?;
    }
    println!("Saved mask to {:?} ({} values)", mask_path, mask_vec.len());

    // Also save shape info
    let shape_path = out_dir.join("embed_shape.txt");
    let (batch, seq, hidden) = embeds.dims3()?;
    std::fs::write(&shape_path, format!("{} {} {}", batch, seq, hidden))?;
    println!("Saved shape: {} x {} x {}", batch, seq, hidden);

    println!("\nDone! Use these embeddings in Python script.");

    Ok(())
}
