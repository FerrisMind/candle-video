//! Stable Video Diffusion (SVD) integration for Candle
//!
//! This module provides image-to-video generation using the SVD architecture.
//! Key components:
//! - CLIP image encoder for conditioning
//! - UNet with spatio-temporal attention blocks
//! - Temporal VAE decoder
//! - Euler discrete scheduler with v-prediction

pub mod clip;
pub mod config;
pub mod pipeline;
pub mod scheduler;
pub mod unet;
pub mod vae;
pub mod weights;

// Re-exports
pub use clip::{ClipVisionModelWithProjection, normalize_for_clip};
pub use config::{
    ClipEncoderConfig, EulerSchedulerConfig, SvdConfig, SvdInferenceConfig, SvdUnetConfig,
    SvdVaeConfig,
};
pub use pipeline::{SvdPipeline, load_image, save_video_frames};
pub use scheduler::EulerDiscreteScheduler;
pub use unet::{
    CrossAttnDownBlockSpatioTemporal, CrossAttnUpBlockSpatioTemporal, DownBlockSpatioTemporal,
    SpatioTemporalResBlock, TransformerSpatioTemporalModel, UNetMidBlockSpatioTemporal,
    UNetSpatioTemporalConditionModel, UpBlockSpatioTemporal,
};
pub use vae::{AutoencoderKLTemporalDecoder, TemporalDecoder};
pub use weights::{WeightLoader, load_safetensors, load_sharded_safetensors};
