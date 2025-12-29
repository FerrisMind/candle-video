//! # Candle Video
//!
//! Video generation models for the Candle framework.
//!
//! This crate provides implementations for:
//! - **LTX-Video**: Text-to-video generation using DiT architecture
//! - **SVD (Stable Video Diffusion)**: Image-to-video using UNet spatio-temporal architecture

pub mod ltxv;
pub mod svd;

// Re-export LTX-Video components for backward compatibility
pub use ltxv::*;
