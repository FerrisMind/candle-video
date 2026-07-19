//! Optional end-to-end Wan pipeline smoke tests (requires local Diffusers weights).

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::{
    WanDenoiseStack, WanGenerateRequest, WanPipeline, latent_shape_from_config, load_wan_config,
};

use wan_fixtures::{wan_consolidated_root, wan_diffusers_root};

fn wan_model_root() -> Option<PathBuf> {
    wan_consolidated_root().or_else(wan_diffusers_root)
}

fn transformer_fixture_embeds() -> Option<Tensor> {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wan/transformer_forward_tiny.json");
    if !fixture.exists() {
        return None;
    }
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(&fixture).ok()?).ok()?;
    let shape: Vec<usize> = fixture["shapes"]["encoder_hidden_states"]
        .as_array()?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let data: Vec<f32> = fixture["encoder_hidden_states"]
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let device = Device::Cpu;
    Some(Tensor::from_vec(data, shape.as_slice(), &device).expect("embeds"))
}

#[test]
fn latent_shape_from_config_matches_diffusers_if_weights_present() {
    let root = match wan_model_root() {
        Some(p) => p,
        None => {
            eprintln!("Skipping latent shape: weights not found");
            return;
        }
    };

    let config = load_wan_config(&root).expect("config");
    assert_eq!(
        latent_shape_from_config(&config, 1, 32, 32, 5),
        vec![1, 16, 2, 4, 4]
    );
}

#[test]
fn denoise_decode_smoke_with_fixture_embeds_if_weights_present() {
    let root = match wan_model_root() {
        Some(p) => p,
        None => {
            eprintln!("Skipping denoise smoke: weights not found");
            return;
        }
    };
    let prompt_embeds = match transformer_fixture_embeds() {
        Some(v) => v,
        None => {
            eprintln!("Skipping denoise smoke: transformer fixture missing");
            return;
        }
    };

    let device = Device::Cpu;
    let mut stack =
        WanDenoiseStack::load(&root, &device, DType::F32, DType::F32).expect("load denoise stack");

    let out = stack
        .denoise_and_decode(32, 32, 1, 2, 1.0, Some(42), None, prompt_embeds, None)
        .expect("denoise+decode");

    let dims = out.frames.dims();
    assert_eq!(dims.len(), 5);
    assert_eq!(dims[0], 1);
    assert_eq!(dims[3], 32);
    assert_eq!(dims[4], 32);
}

#[test]
#[ignore = "loads UMT5; run with --ignored --test wan_pipeline_smoke"]
fn wan_pipeline_full_smoke_if_weights_present() {
    let root = match wan_model_root() {
        Some(p) => p,
        None => {
            eprintln!("Skipping full pipeline smoke: weights not found");
            return;
        }
    };

    let device = Device::Cpu;
    let mut pipeline =
        WanPipeline::load(&root, &device, DType::F32, DType::F32).expect("load wan pipeline");

    let req = WanGenerateRequest {
        prompt: Some("a cat".to_string()),
        negative_prompt: Some(String::new()),
        height: 32,
        width: 32,
        num_frames: 5,
        num_inference_steps: 2,
        guidance_scale: 1.0,
        seed: Some(42),
        max_sequence_length: 226,
        initial_latents: None,
        prompt_embeds: None,
        negative_prompt_embeds: None,
    };

    let out = pipeline.generate(req).expect("generate");
    assert_eq!(out.frames.dims()[2], 5);
}
