//! Candle-Video: LTX-Video integration for Candle framework.
//!
//! This crate provides Rust implementations of video generation models,
//! specifically LTX-Video and Stable Video Diffusion.

pub mod models;
pub mod utils;
pub(crate) mod interfaces;

// LTX Video exports (primary API)
pub use models::ltx_video::{
    loader, ltx_transformer, scheduler, vae, t2v_pipeline,
    configs, quantized_t5_encoder, text_encoder, weight_format,
};

// SVD exports (under svd namespace to avoid conflicts)
pub use models::svd;
