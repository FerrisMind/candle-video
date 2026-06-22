//! Consolidated Wan bundle loads transformer + VAE from native safetensors.

mod wan_fixtures;

use candle_core::{DType, Device};
use candle_video::models::wan::{
    AutoencoderKLWan, WanFullConfig, WanLayout, WanTransformer3DModel, detect_wan_layout,
};

use wan_fixtures::wan_consolidated_root;

#[test]
fn consolidated_transformer_and_vae_load() {
    let root = match wan_consolidated_root() {
        Some(r) => r,
        None => {
            eprintln!("Skipping: consolidated Wan bundle not found");
            return;
        }
    };

    let layout = detect_wan_layout(&root).expect("layout");
    assert!(matches!(layout, WanLayout::Consolidated(_)));

    let config = WanFullConfig::wan21_t2v_13b();
    let device = Device::Cpu;

    let transformer =
        WanTransformer3DModel::load_with_layout(&layout, &device, DType::F32, &config.transformer)
            .expect("transformer");
    assert_eq!(transformer.config().num_layers, 30);

    let vae =
        AutoencoderKLWan::load_with_layout(&layout, &config.vae, &device, DType::F32).expect("vae");
    assert_eq!(vae.config().z_dim, 16);
}

#[test]
fn consolidated_gguf_text_encoder_load() {
    let root = match wan_consolidated_root() {
        Some(r) => r,
        None => return,
    };
    let layout = detect_wan_layout(&root).expect("layout");
    let device = Device::Cpu;
    let te =
        candle_video::models::wan::Umt5TextEncoder::load_for_layout(&layout, &device, DType::F16)
            .expect("gguf te");
    assert!(te.is_quantized());
    assert_eq!(te.config().vocab_size, 256_384);
}
