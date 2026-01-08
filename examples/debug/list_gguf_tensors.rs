use candle_core::quantized::gguf_file;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: list_gguf_tensors <path_to_gguf>");
        return Ok(());
    }
    let path = &args[1];
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;

    println!("Tensors in {}:", path);
    let mut names: Vec<_> = content.tensor_infos.keys().collect();
    names.sort();

    for name in names {
        let info = &content.tensor_infos[name];
        println!("{}: {:?} ({:?})", name, info.shape, info.ggml_dtype);
    }

    Ok(())
}
