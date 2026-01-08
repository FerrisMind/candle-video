use candle_core::safetensors::load;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    local_weights: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let weights_path =
        PathBuf::from(&args.local_weights).join("vae/diffusion_pytorch_model.safetensors");

    println!("Loading weights from {:?}", weights_path);
    let tensors = load(&weights_path, &candle_core::Device::Cpu)?;

    // Look for decoder.scale_shift_table and decoder.time_embedder
    let mut decoder_keys: Vec<_> = tensors
        .keys()
        .filter(|k| {
            k.starts_with("decoder.")
                && !k.contains("mid_block")
                && !k.contains("up_blocks")
                && !k.contains("conv")
        })
        .collect();
    decoder_keys.sort();

    println!("\n--- Decoder-level Keys (excluding mid_block/up_blocks/conv) ---");
    for k in decoder_keys {
        let t = tensors.get(k).unwrap();
        println!("{}: {:?}", k, t.dims());
    }

    Ok(())
}
