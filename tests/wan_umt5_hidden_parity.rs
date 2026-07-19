//! UMT5 embedding layer parity (before transformer blocks).

mod wan_fixtures;

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::Umt5TextEncoder;

use wan_fixtures::wan_diffusers_root;

fn weights_root() -> Option<PathBuf> {
    wan_diffusers_root()
}

#[test]
fn umt5_embed_tokens_match_hf_if_weights_present() {
    let root = match weights_root() {
        Some(p) => p,
        None => return,
    };

    let ref_first8 = [
        -3.731_552_f32,
        -6.049_055_6_f32,
        -0.106_356_23_f32,
        -1.574_433_f32,
        -0.394_854_84_f32,
        3.178_080_6_f32,
        0.421_798_35_f32,
        -2.199_708_7_f32,
    ];

    let device = Device::Cpu;
    let dir = root.join("text_encoder");
    let mut model = Umt5TextEncoder::load(&dir, &device, DType::F32).expect("load custom UMT5");

    // ids for "A cat walking in the snow"
    let ids = Tensor::new(&[320u32, 6283, 53049, 301, 312, 45540, 1], &device)
        .expect("ids")
        .reshape((1, 7))
        .expect("shape");

    // Access shared embedding via forward on ids - compare first token embed indirectly
    let hidden = model.forward_hidden_states(&ids).expect("forward");
    let got: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();
    eprintln!("hidden first8 {:?}", &got[..8]);
    eprintln!("ref   first8 {:?}", ref_first8);

    assert_eq!(hidden.dims(), &[1, 7, 4096]);
    let hf_hidden_first8 = [
        0.0016440969_f32,
        -0.05930363,
        -0.029718762,
        0.0025027555,
        -0.057887416,
        0.056508403,
        -0.05384075,
        0.16152528,
    ];
    let max = hf_hidden_first8
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("hidden max diff first8: {max}");
    assert!(max < 0.02, "UMT5 first-token FP8 error too high: {max}");
}

#[test]
fn scaled_fp8_loader_is_used_for_local_checkpoint_if_present() {
    let root = match weights_root() {
        Some(p) => p,
        None => return,
    };
    let path = root
        .join("text_encoder")
        .join("umt5_xxl_fp8_e4m3fn_scaled.safetensors");
    if !path.exists() {
        return;
    }
    let device = Device::Cpu;
    let encoder = Umt5TextEncoder::load(root.join("text_encoder"), &device, DType::F16)
        .expect("scaled FP8 UMT5 must load through the native backend");
    assert!(!encoder.is_quantized());
    assert_eq!(encoder.config().d_model, 4096);
}
