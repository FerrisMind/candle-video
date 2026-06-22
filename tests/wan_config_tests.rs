//! Wan Diffusers config and weight discovery tests.

use candle_video::models::wan::{
    WanComponentPaths, WanVariant, WanWeightInventory, discover_safetensors, load_wan_config,
    validate_model_index,
};
use std::path::PathBuf;

fn wan_fixture_root() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Wan2.1-T2V-1.3B-Diffusers"),
        PathBuf::from("/home/mod479711/Downloads/models/Wan2.1-T2V-1.3B-Diffusers"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Wan2.1-T2V-1.3B-Diffusers"),
        PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("model_index.json").exists())
}

fn wan_consolidated_root() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Wan2.1-T2V-1.3B"),
        PathBuf::from("/home/mod479711/Downloads/models/Wan2.1-T2V-1.3B"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("text_encoder_gguf").exists())
}

#[test]
fn wan_model_index_validation() {
    let root = match wan_fixture_root() {
        Some(r) => r,
        None => {
            eprintln!("Skipping wan_model_index_validation: fixture not found");
            return;
        }
    };

    let paths = WanComponentPaths::from_root(&root);
    paths.validate_exists().expect("fixture paths");

    let config = load_wan_config(&root).expect("load_wan_config");
    assert_eq!(config.variant, WanVariant::Wan21T2v13B);
    validate_model_index(&config.model_index).expect("index valid");

    assert_eq!(config.transformer.num_layers, 30);
    assert_eq!(config.transformer.num_attention_heads, 12);
    assert_eq!(config.transformer.in_channels, 16);
    assert_eq!(config.transformer.patch_size, [1, 2, 2]);
    assert_eq!(config.vae.z_dim, 16);
    assert_eq!(config.scheduler.flow_shift, 3.0);
    assert_eq!(config.text_encoder.d_model, 4096);
    assert!(config.variant.is_implemented());
}

#[test]
fn wan_weight_inventory_discovery() {
    let root = match wan_fixture_root() {
        Some(r) => r,
        None => {
            eprintln!("Skipping wan_weight_inventory_discovery: fixture not found");
            return;
        }
    };

    let inventory = WanWeightInventory::discover(&root).expect("discover weights");
    assert!(
        inventory.has_all_components(),
        "expected transformer, vae, and text_encoder shards: {:?}",
        inventory
    );

    let transformer_shards = discover_safetensors(&root.join("transformer")).expect("transformer");
    assert!(!transformer_shards.is_empty());
}

#[test]
fn wan_consolidated_layout_discovery() {
    let root = match wan_consolidated_root() {
        Some(r) => r,
        None => {
            eprintln!("Skipping wan_consolidated_layout_discovery: bundle not found");
            return;
        }
    };

    use candle_video::models::wan::{WanLayout, WanWeightInventory, detect_wan_layout};
    let layout = detect_wan_layout(&root).expect("detect layout");
    assert!(matches!(layout, WanLayout::Consolidated(_)));

    let inventory = WanWeightInventory::discover(&root).expect("discover");
    assert!(inventory.has_all_components());
    assert!(inventory.text_encoder_gguf.is_some());
}

#[test]
fn engine_capabilities_wan21() {
    use candle_video::ModelCapabilities;
    let caps = ModelCapabilities::wan21_t2v_13b();
    assert_eq!(caps.model_id, "wan:2.1-t2v-1.3b");
    assert!(caps.supports_negative_prompt);
    assert!(caps.supports_quantized_text_encoder);
    assert!(!caps.supports_two_stage_denoising);
}
