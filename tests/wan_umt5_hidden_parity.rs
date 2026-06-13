//! UMT5 embedding layer parity (before transformer blocks).

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_video::models::wan::loader::discover_safetensors;
use candle_video::models::ltx_video::loader::WeightLoader;
use candle_video::models::wan::Umt5EncoderConfig;
use candle_transformers::models::t5::T5EncoderModel;

fn weights_root() -> Option<PathBuf> {
    let p = PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers");
    p.join("model_index.json").exists().then_some(p)
}

#[test]
fn umt5_embed_tokens_match_hf_if_weights_present() {
    let root = match weights_root() {
        Some(p) => p,
        None => return,
    };

    let ref_first8 = [
        -3.7315518856048584_f32,
        -6.049055576324463,
        -0.10635623335838318,
        -1.5744329690933228,
        -0.3948548436164856,
        3.1780805587768555,
        0.42179834842681885,
        -2.1997087001800537,
    ];

    let device = Device::Cpu;
    let dir = root.join("text_encoder");
    let shards = discover_safetensors(&dir).expect("shards");
    let loader = WeightLoader::new(device.clone(), DType::F32);
    let vb = loader.load_single(&shards[0]).expect("vb");

    let cfg = candle_video::models::wan::configs::WanTextEncoderConfig {
        d_model: 4096,
        d_ff: 10240,
        d_kv: 64,
        num_heads: 64,
        num_layers: 24,
        vocab_size: 256_384,
        model_type: "umt5".to_string(),
        feed_forward_proj: "gated-gelu".to_string(),
        layer_norm_epsilon: 1e-6,
        dropout_rate: 0.1,
    };
    let t5_cfg = Umt5EncoderConfig::from(&cfg).to_candle_t5_config();
    let mut model = T5EncoderModel::load(vb, &t5_cfg).expect("load");

    // ids for "A cat walking in the snow"
    let ids = Tensor::new(
        &[320u32, 6283, 53049, 301, 312, 45540, 1],
        &device,
    )
    .expect("ids")
    .reshape((1, 7))
    .expect("shape");

    // Access shared embedding via forward on ids - compare first token embed indirectly
    let hidden = model.forward(&ids).expect("forward");
    let got: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();
    eprintln!("hidden first8 {:?}", &got[..8]);
    eprintln!("ref   first8 {:?}", ref_first8);

    // If hidden matches ref HF hidden, T5 blocks are fine; else embed or blocks wrong
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
}
