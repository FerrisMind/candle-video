//! Patch embedding parity at production spatial size.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};

use candle_nn::VarBuilder;
use candle_video::models::ltx_video::loader::WeightLoader;
use candle_video::models::wan::WanPatchEmbedding;
use candle_video::models::wan::loader::discover_safetensors;
use wan_fixtures::wan_transformer_dir;

#[test]
fn patch_embedding_production_shape_matches_fixture() {
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/wan/step0_reference.json"),
        )
        .expect("read"),
    )
    .expect("json");

    let lat_shape: Vec<usize> = fixture["latents_init_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let lat_data: Vec<f32> = fixture["latents_init"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let device = Device::Cpu;
    let latents = Tensor::from_vec(lat_data, lat_shape.as_slice(), &device).expect("latents");

    let dir = match wan_transformer_dir() {
        Some(p) => p,
        None => return,
    };
    let shards = discover_safetensors(&dir).expect("shards");
    let loader = WeightLoader::new(device.clone(), DType::F32);
    let vb = loader.load_sharded(shards.as_slice()).expect("vb");
    let patch =
        WanPatchEmbedding::new(16, 1536, [1, 2, 2], vb.pp("patch_embedding")).expect("patch");

    let out = patch.forward(&latents).expect("forward");
    let row: Vec<f32> = out
        .i((0, 0, ..))
        .expect("row")
        .to_vec1()
        .expect("vec")
        .into_iter()
        .take(4)
        .collect();

    let expected = [
        -0.2295476645231247f32,
        0.1298161745071411,
        0.045674264430999756,
        -0.26436924934387207,
    ];
    for (i, (&e, &g)) in expected.iter().zip(row.iter()).enumerate() {
        eprintln!("patch dim {i}: ref={e} got={g} diff={}", (e - g).abs());
    }
    let max_diff = expected
        .iter()
        .zip(row.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-4, "patch embed max diff {max_diff}");
}
