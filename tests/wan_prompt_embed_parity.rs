//! Prompt embedding parity vs Diffusers reference.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_video::models::wan::{Umt5TextEncoder, WanTokenizer, encode_prompt};

use wan_fixtures::wan_diffusers_root;

fn weights_root() -> Option<PathBuf> {
    wan_diffusers_root()
}

fn ref_path() -> Option<PathBuf> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/prompt_embeds_ref.json");
    p.exists().then_some(p)
}

#[test]
fn prompt_embeds_match_diffusers_if_fixture_present() {
    let root = match weights_root() {
        Some(p) => p,
        None => return,
    };
    let ref_path = match ref_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: missing tests/fixtures/wan/prompt_embeds_ref.json");
            return;
        }
    };

    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_path).expect("read")).expect("json");

    let device = Device::Cpu;
    let tokenizer = WanTokenizer::from_pretrained(root.join("tokenizer")).expect("tok");
    let mut text_encoder =
        Umt5TextEncoder::load(root.join("text_encoder"), &device, DType::F32).expect("te");

    let prompt = vec!["A cat walking in the snow".to_string()];
    let neg = vec![String::new()];
    let (p, n) = encode_prompt(
        &tokenizer,
        &mut text_encoder,
        &prompt,
        Some(neg.as_slice()),
        true,
        226,
        &device,
    )
    .expect("encode");

    let got: Vec<f32> = p.flatten_all().unwrap().to_vec1().unwrap();
    let ref64: Vec<f32> = fixture["prompt_first64"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let max_diff = ref64
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    eprintln!("prompt embed max diff first64: {max_diff}");
    eprintln!("rust first8: {:?}", &got[..8]);
    eprintln!("ref  first8: {:?}", &ref64[..8]);

    let neg_got: Vec<f32> = n
        .as_ref()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let ref_neg: Vec<f32> = fixture["neg_first64"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let neg_max = ref_neg
        .iter()
        .zip(neg_got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("neg embed max diff first64: {neg_max}");

    assert!(max_diff < 1e-3, "prompt embeds diverge: {max_diff}");
    assert!(neg_max < 1e-3, "neg embeds diverge: {neg_max}");
}
