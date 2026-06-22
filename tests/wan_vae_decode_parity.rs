//! Wan 2.1 VAE decode parity tests.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::vae::denormalize_latents_vec;
use candle_video::models::wan::{AutoencoderKLWan, WanVaeConfig};

use wan_fixtures::wan_vae_dir;

fn wan_vae_fixture_dir() -> Option<PathBuf> {
    wan_vae_dir().filter(|p| p.join("diffusion_pytorch_model.safetensors").exists())
}

#[test]
fn latent_denorm_math_matches_pipeline_formula() {
    let mean = vec![-0.7571_f32, 0.9653];
    let std = vec![2.8184_f32, 1.7708];
    let z = vec![1.0_f32, 2.0];
    let out = denormalize_latents_vec(&z, &mean, &std);
    assert!((out[0] - (1.0 * 2.8184 - 0.7571)).abs() < 1e-4);
    assert!((out[1] - (2.0 * 1.7708 + 0.9653)).abs() < 1e-4);
}

#[test]
fn autoencoder_denormalize_tensor_shape() {
    let device = Device::Cpu;
    let cfg = WanVaeConfig {
        base_dim: 96,
        dim_mult: vec![1, 2, 4, 4],
        dropout: 0.0,
        latents_mean: vec![-0.7571; 16],
        latents_std: vec![2.8184; 16],
        num_res_blocks: 2,
        temporal_downsample: vec![false, true, true],
        z_dim: 16,
        attn_scales: vec![],
    };
    let mean = Tensor::from_vec(cfg.latents_mean.clone(), (1, 16, 1, 1, 1), &device).unwrap();
    let std = Tensor::from_vec(cfg.latents_std.clone(), (1, 16, 1, 1, 1), &device).unwrap();
    let z = Tensor::ones((1, 16, 1, 4, 4), DType::F32, &device).unwrap();
    let denorm = z.broadcast_mul(&std).unwrap().broadcast_add(&mean).unwrap();
    assert_eq!(denorm.dims(), &[1, 16, 1, 4, 4]);
}

#[test]
fn load_wan_vae_from_local_fixture_if_present() {
    let vae_dir = match wan_vae_fixture_dir() {
        Some(p) => p,
        None => {
            eprintln!("Skipping load_wan_vae_from_local_fixture_if_present: fixture not found");
            return;
        }
    };

    let device = Device::Cpu;
    let vae = AutoencoderKLWan::load(&vae_dir, &device, DType::F32).expect("load vae");
    assert_eq!(vae.config().z_dim, 16);
    assert_eq!(vae.config().base_dim, 96);
}

#[test]
fn decode_tiny_random_latent_if_fixture_present() {
    let vae_dir = match wan_vae_fixture_dir() {
        Some(p) => p,
        None => {
            eprintln!("Skipping decode_tiny_random_latent_if_fixture_present: fixture not found");
            return;
        }
    };

    let device = Device::Cpu;
    let mut vae = AutoencoderKLWan::load(&vae_dir, &device, DType::F32).expect("load vae");

    // One latent frame at 4x4 spatial (tiny) — exercises full decode graph.
    let latent = Tensor::randn(0f32, 1.0, (1, 16, 1, 4, 4), &device).expect("latent");
    let decoded = vae.decode(&latent).expect("decode");
    assert_eq!(decoded.dims(), &[1, 3, 1, 32, 32]);
    let flat = decoded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|v| (-1.0..=1.0).contains(v)));
}
