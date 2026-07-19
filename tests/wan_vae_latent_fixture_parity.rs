//! Decode Diffusers final latents through Rust VAE and compare to reference stats.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::AutoencoderKLWan;

use wan_fixtures::wan_vae_dir;

fn vae_dir() -> Option<PathBuf> {
    wan_vae_dir().filter(|p| p.join("diffusion_pytorch_model.safetensors").exists())
}

fn fixture_path() -> Option<PathBuf> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/vae_decode_tiny.json");
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
    let shape: Vec<usize> = fixture["latent_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let data: Vec<f32> = fixture["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let device = Device::Cpu;
    let mut vae = AutoencoderKLWan::load(&vae_dir, &device, DType::F32).expect("load vae");
    let latents = Tensor::from_vec(data, shape.as_slice(), &device).expect("latents");
    let decoded = vae.decode(&latents).expect("decode");

    let expected_shape: Vec<usize> = fixture["decoded_shape"]
        .as_array()
        .expect("decoded shape")
        .iter()
        .map(|v| v.as_u64().expect("shape value") as usize)
        .collect();
    assert_eq!(decoded.dims(), expected_shape.as_slice());
    let expected: Vec<f32> = fixture["decoded"]
        .as_array()
        .expect("decoded values")
        .iter()
        .map(|v| v.as_f64().expect("decoded value") as f32)
        .collect();

    let flat = decoded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(flat.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_rust = 0.0f64;
    let mut norm_ref = 0.0f64;
    for (actual, reference) in flat.iter().zip(expected.iter()) {
        let diff = (*actual - *reference).abs();
        max_abs = max_abs.max(diff);
        mean_abs += diff;
        dot += *actual as f64 * *reference as f64;
        norm_rust += *actual as f64 * *actual as f64;
        norm_ref += *reference as f64 * *reference as f64;
    }
    mean_abs /= flat.len() as f32;
    let cosine = dot / (norm_rust.sqrt() * norm_ref.sqrt());
    eprintln!(
        "VAE decode parity: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cosine:.9}"
    );
    assert!(max_abs < 5e-3, "VAE max abs error too high: {max_abs:.6e}");
    assert!(
        mean_abs < 5e-4,
        "VAE mean abs error too high: {mean_abs:.6e}"
    );
    assert!(cosine > 0.9999, "VAE cosine too low: {cosine:.9}");

    let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let var = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    let std = var.sqrt();

    eprintln!(
        "rust vae on diffusers latents: min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}"
    );
    assert!(std.is_finite() && mean.is_finite() && min.is_finite() && max.is_finite());
}
