//! Wan transformer block 0 parity vs Diffusers reference dump.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::ltx_video::loader::WeightLoader;
use candle_video::models::wan::loader::discover_safetensors;
use candle_video::models::wan::{WanRotaryPosEmbed, WanTransformerBlock, WanTransformerConfig};

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
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/transformer_block0.json");
    path.exists().then_some(path)
}

fn load_tensor(data: &[f32], shape: &[usize], device: &Device) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, device).expect("tensor")
}

#[test]
fn load_transformer_block0_from_local_fixture_if_present() {
    let dir = match transformer_fixture_dir() {
        Some(p) => p,
        None => {
            eprintln!("Skipping load_transformer_block0: transformer weights not found");
            return;
        }
    };

    let device = Device::Cpu;
    let cfg: WanTransformerConfig = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).expect("config"),
    )
    .expect("parse config");

    let shards = discover_safetensors(&dir).expect("shards");
    let loader = WeightLoader::new(device.clone(), DType::F32);
    let vb = if shards.len() == 1 {
        loader.load_single(&shards[0]).expect("vb")
    } else {
        loader.load_sharded(shards.as_slice()).expect("vb")
    };

    let _block = WanTransformerBlock::new(&cfg, vb.pp("blocks").pp("0")).expect("block");
}

#[test]
fn transformer_block0_forward_matches_reference_if_fixture_present() {
    let fixture_path = match fixture_json() {
        Some(p) => p,
        None => {
            eprintln!(
                "Skipping transformer_block0_forward_matches_reference: run scripts/wan/reference/dump_transformer_block.py"
            );
            return;
        }
    };
    let dir = match transformer_fixture_dir() {
        Some(p) => p,
        None => {
            eprintln!("Skipping transformer_block0_forward: weights not found");
            return;
        }
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture_path).expect("read fixture")).expect("json");

    let device = Device::Cpu;
    let cfg: WanTransformerConfig = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).expect("config"),
    )
    .expect("parse config");

    let shards = discover_safetensors(&dir).expect("shards");
    let loader = WeightLoader::new(device.clone(), DType::F32);
    let vb = if shards.len() == 1 {
        loader.load_single(&shards[0]).expect("vb")
    } else {
        loader.load_sharded(shards.as_slice()).expect("vb")
    };
    let block = WanTransformerBlock::new(&cfg, vb.pp("blocks").pp("0")).expect("block");
    let rope = WanRotaryPosEmbed::new(&cfg, &device).expect("rope");

    let hs_shape: Vec<usize> = fixture["shapes"]["hidden_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let enc_shape: Vec<usize> = fixture["shapes"]["encoder_hidden_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let temb_shape: Vec<usize> = fixture["shapes"]["timestep_proj"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let latent_shape: Vec<usize> = fixture["shapes"]["latent_for_rope"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let out_shape: Vec<usize> = fixture["shapes"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let to_f32 = |key: &str| -> Vec<f32> {
        fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };

    let hidden_states = load_tensor(&to_f32("hidden_states"), &hs_shape, &device);
    let encoder_hidden_states = load_tensor(&to_f32("encoder_hidden_states"), &enc_shape, &device);
    let timestep_proj = load_tensor(&to_f32("timestep_proj"), &temb_shape, &device);
    let latent = load_tensor(&to_f32("latent_for_rope"), &latent_shape, &device);
    let expected = load_tensor(&to_f32("output"), &out_shape, &device);

    let latent_5d = latent.reshape(latent_shape).expect("latent reshape");
    let (cos, sin) = rope.forward(&latent_5d).expect("rope");

    let out = block
        .forward(
            &hidden_states,
            &encoder_hidden_states,
            &timestep_proj,
            (&cos, &sin),
        )
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
        diff < 1e-3,
        "block0 max abs diff {diff} exceeds tolerance (expected < 1e-3)"
    );
}
