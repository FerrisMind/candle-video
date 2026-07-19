//! Candle-Video: Rust-native video generation for the Candle framework.
//!
//! Supports LTX-Video today; Wan 2.1 T2V and additional backends are added
//! behind the unified [`engine`] module.

pub mod engine;
pub mod models;
pub mod profiling;
pub mod utils;

pub use engine::{
    CancellationToken, DenoisePass, GenerateRequest, GenerationEvent, GenerationStage,
    LtxPipelineAdapter, MemoryOptions, ModelCapabilities, ModelRegistry, NoopProgressObserver,
    ProgressObserver, RunManifest, VideoModelSpec, VideoOutput, VideoPipeline, VideoTask,
    WanDevicePlan, WanPipelineAdapter, ensure_not_cancelled, ensure_wan_cuda_requirements,
    resolve_seed, run_with_heartbeat, wan_patch_token_count, write_run_manifest,
};
pub use models::ltx_video::*;
pub use models::wan::{
    WanDenoiseStack, WanGenerateRequest, WanPipeline, WanPipelineOutput, WanSchedulerProfile,
    latent_shape_from_config, write_wan_manifest,
};
