//! GGUF UMT5 prompt embeddings vs full-precision Diffusers reference.

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_video::models::wan::loader::WanConsolidatedPaths;
use candle_video::models::wan::{Umt5TextEncoder, WanTokenizer, encode_prompt};

use wan_fixtures::wan_consolidated_root;

fn ref_path() -> Option<PathBuf> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wan/prompt_embeds_ref.json");
    p.exists().then_some(p)
}

#[test]
fn gguf_prompt_embeds_within_quantization_tolerance() {
    let root = match wan_consolidated_root() {
        Some(r) => r,
        None => {
            eprintln!("Skipping: consolidated bundle not found");
            return;
        }
    };
    let ref_path = match ref_path() {
        Some(p) => p,
        None => return,
    };
    let paths = WanConsolidatedPaths::from_root(&root).expect("paths");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_path).expect("read")).expect("json");

    let device = Device::Cpu;
    let tokenizer =
        WanTokenizer::from_pretrained(&paths.tokenizer_json.parent().unwrap()).expect("tok");
    let mut text_encoder =
        Umt5TextEncoder::load_gguf(&paths.text_encoder_gguf, &device).expect("gguf te");
    assert!(text_encoder.is_quantized());

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

    eprintln!("GGUF prompt embed max diff first64: {max_diff}");
    // ponytail: Q5_K_M UMT5 — same ballpark as LTX T5 GGUF (~0.8 ceiling in verify_t5)
    assert!(
        max_diff < 0.85,
        "GGUF prompt embeds diverge too much from FP32 ref: {max_diff}"
    );

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
    eprintln!("GGUF neg embed max diff first64: {neg_max}");
    assert!(neg_max < 0.85, "GGUF neg embeds diverge: {neg_max}");
}
