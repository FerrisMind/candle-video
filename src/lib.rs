//! # Candle Video
//!
//! Integration of LTX-Video model for the Candle framework.
//!
//! This crate provides components for text-to-video generation using
//! the LTX-Video architecture, including:
//!
//! - Safetensors weight loading with sharding and name mapping
//! - T5 text encoder integration
//! - Diffusion Transformer (DiT) blocks
//! - Causal Video VAE
//! - Rectified Flow scheduler
//! - Text-to-video pipeline

pub mod config;
pub mod dit;
pub mod latents_bin;
pub mod loader;
pub mod pipeline;
pub mod quantized_t5_encoder;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod vae;

// Config exports
pub use config::{DitConfig, InferenceConfig, SchedulerConfig, VaeConfig};

// Loader exports
pub use loader::{
    LoaderError, SafetensorsIndex, TensorInfo, WeightLoader, find_sharded_files, list_tensor_names,
    load_model_config, validate_tensor_names,
};

// DiT exports
pub use dit::{
    AdaLayerNormSingle, Attention, BasicTransformerBlock, CaptionProjection, DiTBlock, FeedForward,
    GELUProj, JointAttention, ModulationLayer, PatchEmbedding, QKNorm, RMSNorm, RMSNormNoWeight,
    SkipLayerStrategy, TimestepEmbedding, Transformer3DModel,
};

// RoPE exports
pub use rope::{FractionalRoPE, generate_coord_grids, generate_indices_grid};

// Pipeline exports
pub use pipeline::{
    ConditioningItem, PipelineConfig, TextToVideoPipeline, add_noise_to_conditioning_latents,
    apply_cfg_star_rescale, apply_conditioning_mask, compute_token_timesteps, tone_map_latents,
};

// Latents bin IO exports
pub use latents_bin::{read_f32_tensor_with_header, write_f32_tensor_with_header};

// Scheduler exports
pub use scheduler::{RectifiedFlowScheduler, SchedulerOutput};

// Text encoder exports
pub use text_encoder::{
    QuantizedT5Encoder, T5EncoderConfig, T5TextEncoderWrapper, TextEncoderError,
};

// Quantized T5 encoder exports
pub use quantized_t5_encoder::QuantizedT5EncoderModel;

// VAE exports
pub use vae::{
    CausalConv3d, LatentLogVar, RMSNorm3d, ResBlock, ResBlockGroup, SpaceToDepthDownsample,
    TimeEmbedder, Upsampler, VaeDecoder, VaeEncoder, causal_pad, patchify, spatial_pad, unpatchify,
    vae_encode,
};

