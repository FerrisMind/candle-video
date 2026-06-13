//! Decode Diffusers final latents through Rust VAE and compare to reference stats.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::AutoencoderKLWan;

fn vae_dir() -> Option<PathBuf> {
    let p = PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers/vae");
    p.join("diffusion_pytorch_model.safetensors")
        .exists()
        .then_some(p)
}

fn fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/final_latents_ref.json");
    p.exists().then_some(p)
}

#[test]
fn rust_vae_decode_matches_diffusers_on_shared_latents() {
    let vae_dir = match vae_dir() {
        Some(p) => p,
        None => return,
    };
    let fixture_path = match fixture_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: export final_latents_ref.json first");
            return;
        }
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path).expect("read")).expect("json");
    let shape: Vec<usize> = fixture["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let data: Vec<f32> = fixture["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let device = Device::Cpu;
    let mut vae = AutoencoderKLWan::load(&vae_dir, &device, DType::F32).expect("load vae");
    let latents = Tensor::from_vec(data, shape.as_slice(), &device).expect("latents");
    let decoded = vae.decode(&latents).expect("decode");

    let flat = decoded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let var = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    let std = var.sqrt();

    eprintln!(
        "rust vae on diffusers latents: min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}"
    );
    // Diffusers reference on same latents: std~0.24 mean~0.50
    assert!(
        std < 0.5,
        "VAE output too noisy: std={std:.4} (expected ~0.24)"
    );
    assert!(
        (mean - 0.5).abs() < 0.35,
        "VAE mean off: {mean:.4} (expected ~0.50)"
    );
}
