//! Wan 2.1 / 2.2 video generation backend (Diffusers-compatible).
//!
//! MVP scope: Wan 2.1 T2V 1.3B text-to-video inference.

pub mod configs;
pub mod loader;
pub mod pipeline;
pub mod prompt_clean;
pub mod prompt_encode;
pub mod quantized_umt5_encoder;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod umt5;
pub mod vae;
pub mod weight_format;

pub use configs::validate_model_index;
pub use configs::*;
pub use loader::{
    WanComponentPaths, WanConsolidatedPaths, WanLayout, WanWeightInventory, detect_wan_layout,
    discover_safetensors, load_wan_config, validate_vae_weights, write_wan_manifest,
};
pub use pipeline::{
    WanDenoiseStack, WanGenerateRequest, WanPipeline, WanPipelineOutput, latent_shape_from_config,
};
pub use prompt_clean::*;
pub use prompt_encode::encode_prompt;
pub use scheduler::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig,
    UniPcMultistepScheduler, WanScheduler, WanSchedulerProfile,
};
pub use text_encoder::{Umt5EncoderConfig, Umt5TextEncoder};
pub use tokenizer::{WAN_DEFAULT_MAX_SEQ_LEN, WanTokenizer};
pub use transformer::{
    WanConditionEmbedder, WanPatchEmbedding, WanRotaryPosEmbed, WanTransformer3DModel,
    WanTransformerBlock,
};
pub use vae::AutoencoderKLWan;
