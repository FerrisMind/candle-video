//! Full WanTransformer3DModel forward parity (tiny latent).

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::WanTransformer3DModel;

fn transformer_fixture_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../Wan2.1-T2V-1.3B-Diffusers/transformer"),
        PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers/transformer"),
    ];
    candidates.into_iter().find(|p| {
        p.join("diffusion_pytorch_model.safetensors.index.json").exists()
            || p.join("diffusion_pytorch_model.safetensors").exists()
    })
}

fn fixture_json() -> Option<PathBuf> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/transformer_forward_tiny.json");
    path.exists().then_some(path)
}

fn load_tensor(data: &[f32], shape: &[usize], device: &Device) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, device).expect("tensor")
}

#[test]
fn load_full_wan_transformer_from_local_fixture_if_present() {
    let dir = match transformer_fixture_dir() {
        Some(p) => p,
        None => {
            eprintln!("Skipping load_full_wan_transformer: weights not found");
            return;
        }
    };

    let device = Device::Cpu;
    let model = WanTransformer3DModel::load(&dir, &device, DType::F32).expect("load transformer");
    assert_eq!(model.config().num_layers, 30);
    assert_eq!(model.config().in_channels, 16);
}

#[test]
fn transformer_forward_matches_reference_if_fixture_present() {
    let fixture_path = match fixture_json() {
        Some(p) => p,
        None => {
            eprintln!(
                "Skipping transformer_forward parity: run scripts/wan/reference/dump_transformer_forward.py"
            );
            return;
        }
    };
    let dir = match transformer_fixture_dir() {
        Some(p) => p,
        None => {
            eprintln!("Skipping transformer_forward: weights not found");
            return;
        }
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture_path).expect("read fixture")).expect("json");

    let device = Device::Cpu;
    let model = WanTransformer3DModel::load(&dir, &device, DType::F32).expect("load");

    let to_f32 = |key: &str| -> Vec<f32> {
        fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let shape = |key: &str| -> Vec<usize> {
        fixture["shapes"][key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };

    let hidden_states = load_tensor(&to_f32("hidden_states"), &shape("hidden_states"), &device);
    let timestep = Tensor::from_vec(
        to_f32("timestep").iter().map(|&v| v as i64).collect(),
        shape("timestep"),
        &device,
    )
    .expect("timestep");
    let encoder_hidden_states =
        load_tensor(&to_f32("encoder_hidden_states"), &shape("encoder_hidden_states"), &device);
    let expected = load_tensor(&to_f32("output"), &shape("output"), &device);

    let out = model
        .forward(&hidden_states, &timestep, &encoder_hidden_states)
        .expect("forward");

    assert_eq!(out.dims(), expected.dims());

    let diff = (out.to_dtype(DType::F32).expect("f32") - expected)
        .expect("diff")
        .abs()
        .expect("abs")
        .max_all()
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");

    assert!(
        diff < 5e-3,
        "transformer forward max abs diff {diff} exceeds tolerance (expected < 5e-3)"
    );
}
