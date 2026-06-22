//! Wan tokenizer parity against a Hugging Face reference fixture.

mod wan_fixtures;

use candle_core::Device;
use candle_video::models::wan::{WAN_DEFAULT_MAX_SEQ_LEN, WanTokenizer};
use serde::Deserialize;
use std::path::PathBuf;

use wan_fixtures::wan_diffusers_root;

#[derive(Debug, Deserialize)]
struct TokenizerFixture {
    prompt: String,
    clean_prompt: String,
    max_length: usize,
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    seq_len: usize,
}

fn fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wan/tokenizer_cat_snow.json");
    p.exists().then_some(p)
}

fn wan_tokenizer_dir() -> Option<PathBuf> {
    wan_diffusers_root()
        .map(|r| r.join("tokenizer"))
        .filter(|p| p.join("tokenizer.json").exists())
}

#[test]
fn wan_tokenizer_matches_hf_fixture() {
    let fixture_file = match fixture_path() {
        Some(p) => p,
        None => {
            eprintln!("skip: missing tokenizer fixture");
            return;
        }
    };
    let tokenizer_dir = match wan_tokenizer_dir() {
        Some(p) => p,
        None => {
            eprintln!("skip: Wan tokenizer dir not found");
            return;
        }
    };

    let fixture: TokenizerFixture =
        serde_json::from_str(&std::fs::read_to_string(&fixture_file).unwrap()).unwrap();

    let tokenizer = WanTokenizer::from_pretrained(&tokenizer_dir)
        .expect("load tokenizer")
        .with_device(Device::Cpu);

    let (ids, mask) = tokenizer
        .encode_prompts(std::slice::from_ref(&fixture.prompt), fixture.max_length)
        .expect("encode");

    let ids_vec: Vec<u32> = ids.to_vec2::<u32>().unwrap().pop().unwrap();
    let mask_vec: Vec<u32> = mask
        .to_vec2::<f32>()
        .unwrap()
        .pop()
        .unwrap()
        .into_iter()
        .map(|v| v as u32)
        .collect();

    assert_eq!(ids_vec, fixture.input_ids, "input_ids mismatch");
    assert_eq!(mask_vec, fixture.attention_mask, "attention_mask mismatch");
    assert_eq!(
        mask_vec.iter().map(|&v| v as u32).sum::<u32>() as usize,
        fixture.seq_len
    );
}

#[test]
fn wan_default_max_seq_len_matches_diffusers() {
    assert_eq!(WAN_DEFAULT_MAX_SEQ_LEN, 226);
}
