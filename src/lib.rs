//! Candle-Video: Rust-native video generation for the Candle framework.
//!
//! Supports LTX-Video today; Wan 2.1 T2V and additional backends are added
//! behind the unified [`engine`] module.

pub mod engine;
pub mod models;
pub mod utils;

pub use engine::{
    GenerateRequest, LtxPipelineAdapter, MemoryOptions, ModelCapabilities, ModelRegistry,
    VideoModelSpec, VideoOutput, VideoPipeline, VideoTask, WanPipelineAdapter,
};
pub use models::wan::{WanDenoiseStack, WanGenerateRequest, WanPipeline, WanPipelineOutput};
pub use models::ltx_video::*;
