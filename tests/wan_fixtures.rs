//! Shared paths for Wan integration tests.

// This module is included by several independent integration-test crates.  A
// helper is intentionally used only by the tests that need its checkpoint
// layout, so suppress per-crate dead-code warnings here.
#![allow(dead_code)]

use std::path::PathBuf;

pub fn wan_diffusers_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("CANDLE_VIDEO_WAN_DIFFUSERS_ROOT") {
        candidates.push(PathBuf::from(path));
    }
    candidates.extend([
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Wan2.1-T2V-1.3B-Diffusers"),
        PathBuf::from("/home/mod479711/Downloads/models/Wan2.1-T2V-1.3B-Diffusers"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Wan2.1-T2V-1.3B-Diffusers"),
        PathBuf::from("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers"),
    ]);
    candidates
        .into_iter()
        .find(|p| p.join("model_index.json").exists())
}

pub fn wan_consolidated_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("CANDLE_VIDEO_WAN_MODEL_ROOT") {
        candidates.push(PathBuf::from(path));
    }
    candidates.extend([
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Wan2.1-T2V-1.3B"),
        PathBuf::from("/home/mod479711/Downloads/models/Wan2.1-T2V-1.3B"),
    ]);
    candidates
        .into_iter()
        .find(|p| p.join("text_encoder_gguf").exists())
}

pub fn wan_transformer_dir() -> Option<PathBuf> {
    wan_diffusers_root().map(|r| r.join("transformer"))
}

pub fn wan_vae_dir() -> Option<PathBuf> {
    wan_diffusers_root().map(|r| r.join("vae"))
}

pub fn wan_scheduler_config() -> Option<PathBuf> {
    wan_diffusers_root().map(|r| r.join("scheduler/scheduler_config.json"))
}
