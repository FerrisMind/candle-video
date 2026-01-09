#![allow(unused_imports)]
use candle_core::Device;
use candle_core::safetensors::MmapedSafetensors;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let file = PathBuf::from(
        r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\vae\diffusion_pytorch_model.safetensors",
    );
    let tensors = unsafe { MmapedSafetensors::new(file)? };
    let mut keys: Vec<_> = tensors.tensors().iter().map(|(k, _)| k.clone()).collect();
    keys.sort();
    for k in keys {
        if k.contains("latent") || k.contains("mean") || k.contains("std") || k.len() < 20 {
            println!("{}", k);
        }
    }
    Ok(())
}
