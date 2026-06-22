//! Unified video generation engine abstractions.
//!
//! Backends (LTX, Wan, …) implement [`VideoPipeline`] without sharing model-specific math.

pub mod backends;
pub mod cuda_guard;
pub mod device_plan;
pub mod ltx_adapter;
pub mod memory;
pub mod model;
pub mod pipeline;
pub mod prompt;
pub mod video;
pub mod wan_adapter;

pub use backends::*;
pub use cuda_guard::{require_flash_attn_for_cuda, wan_inference_compute_dtype};
pub use device_plan::{WanDevicePlan, ensure_wan_cuda_requirements, wan_patch_token_count};
pub use ltx_adapter::{LtxPipelineAdapter, build_ltx_pipeline, default_ltx_vae_processor_config};
pub use memory::{MemoryOptions, PrecisionOptions};
pub use model::{EngineLoadOptions, ModelCapabilities, ModelRegistry, VideoModelSpec};
pub use pipeline::{GenerateRequest, VideoPipeline};
pub use prompt::{PromptEmbeds, PromptEncoder};
pub use video::{FrameRule, OutputOptions, Resolution, VideoOutput, VideoTask};
pub use wan_adapter::WanPipelineAdapter;
