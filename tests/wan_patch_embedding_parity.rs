//! Patch embedding conv parity vs PyTorch reference row.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_video::models::ltx_video::loader::WeightLoader;
use candle_video::models::wan::WanPatchEmbedding;
use candle_video::models::wan::loader::discover_safetensors;

use wan_fixtures::wan_transformer_dir;

fn transformer_dir() -> Option<PathBuf> {
    wan_transformer_dir().filter(|p| p.join("config.json").exists())
}

#[test]
fn patch_embedding_row_matches_fixture_reference() {
    let dir = match transformer_dir() {
        Some(p) => p,
        None => return,
    };
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/wan/transformer_forward_tiny.json"),
        )
        .expect("read"),
    )
    .expect("json");

    let device = Device::Cpu;
    let shards = discover_safetensors(&dir).expect("shards");
    let loader = WeightLoader::new(device.clone(), DType::F32);
    let vb = loader.load_single(&shards[0]).expect("vb");

    let pe = WanPatchEmbedding::new(16, 1536, [1, 2, 2], vb.pp("patch_embedding")).expect("pe");

    let shape: Vec<usize> = fixture["shapes"]["hidden_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let data: Vec<f32> = fixture["hidden_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let hs = Tensor::from_vec(data, shape.as_slice(), &device).expect("hs");

    let out = pe.forward(&hs).expect("pe");
    let row = out
        .i((0, 0))
        .expect("row")
        .to_dtype(DType::F32)
        .expect("f32")
        .to_vec1::<f32>()
        .expect("vec");
    let ref_row: Vec<f32> = fixture["intermediates"]["patch_embed_row0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let diff = row
        .iter()
        .zip(ref_row.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < 1e-4,
        "patch row max diff {diff}, rust={:?} ref={:?}",
        &row[..4],
        &ref_row[..4]
    );
}
