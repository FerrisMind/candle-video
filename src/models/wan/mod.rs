//! Wan 2.1 / 2.2 video generation backend (Diffusers-compatible).
//!
//! MVP scope: Wan 2.1 T2V 1.3B text-to-video inference.

pub mod configs;
pub mod loader;
pub mod pipeline;
pub mod prompt_clean;
pub mod prompt_encode;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod umt5;
pub mod transformer;
pub mod vae;

pub use configs::*;
pub use loader::*;
pub use pipeline::{WanGenerateRequest, WanPipeline, WanPipelineOutput, WanDenoiseStack, latent_shape_from_config};
pub use prompt_clean::*;
pub use prompt_encode::encode_prompt;
pub use scheduler::UniPcMultistepScheduler;
pub use text_encoder::{Umt5EncoderConfig, Umt5TextEncoder};
pub use tokenizer::{WanTokenizer, WAN_DEFAULT_MAX_SEQ_LEN};
pub use transformer::{
    WanConditionEmbedder, WanPatchEmbedding, WanRotaryPosEmbed, WanTransformer3DModel,
    WanTransformerBlock,
};
pub use vae::AutoencoderKLWan;
