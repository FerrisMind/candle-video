//! Unified video generation engine abstractions.
//!
//! Backends (LTX, Wan, …) implement [`VideoPipeline`] without sharing model-specific math.

pub mod backends;
pub mod ltx_adapter;
pub mod memory;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod video;
pub mod wan_adapter;

pub use backends::*;
pub use ltx_adapter::{LtxPipelineAdapter, build_ltx_pipeline, default_ltx_vae_processor_config};
pub use wan_adapter::WanPipelineAdapter;
pub use memory::{MemoryOptions, PrecisionOptions};
pub use model::{EngineLoadOptions, ModelCapabilities, ModelRegistry, VideoModelSpec};
pub use pipeline::{GenerateRequest, VideoPipeline};
pub use prompt::{PromptEmbeds, PromptEncoder};
pub use video::{FrameRule, OutputOptions, Resolution, VideoOutput, VideoTask};
