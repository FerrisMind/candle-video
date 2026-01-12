//! Candle-Video: LTX-Video integration for Candle framework.
//!
//! This crate provides Rust implementations of video generation models,
//! specifically LTX-Video and Stable Video Diffusion.

pub mod interfaces;
pub mod models;
pub mod utils;

// LTX Video exports (primary API)
pub use models::ltx_video::{
    configs, loader, ltx_transformer, quantized_t5_encoder, scheduler, t2v_pipeline, text_encoder,
    vae, weight_format,
};

// SVD exports (under svd namespace to avoid conflicts)
pub use models::svd;
