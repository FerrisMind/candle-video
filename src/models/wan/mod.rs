pub mod config;
pub mod loader;
pub mod t2v_pipeline_v2;
pub mod text_encoder;
pub mod transformer_wan;
pub mod vae;

// Legacy pipeline (trait-based, commented out)
// pub mod t2v_pipeline;

// Re-exports - Config
pub use config::{AutoencoderKLWanConfig, WanTransformer3DConfig};

// Re-exports - Models
pub use transformer_wan::WanTransformer3DModel;
pub use vae::{AutoencoderKLWan, DecoderOutput, EncoderOutput};

// Re-exports - Text Encoder
pub use text_encoder::{
    QuantizedUMT5Encoder, UMT5TextEncoder, UMT5Tokenizer, load_quantized_umt5_encoder,
    load_umt5_encoder, load_umt5_encoder_from_file, load_umt5_tokenizer,
};

// Re-exports - Loader
pub use loader::{
    T5EncoderConfig, WanLoaderError, load_quantized_text_encoder, load_text_encoder,
    load_text_encoder_from_file, load_tokenizer, load_tokenizer_from_dir, load_transformer,
    load_transformer_config, load_transformer_from_dir, load_vae, load_vae_config,
    load_vae_from_dir, load_wan_2_1_vae, load_wan_t2v_1_3b_transformer,
    load_wan_t2v_14b_transformer,
};

// Re-exports - Pipeline
pub use t2v_pipeline_v2::{
    OutputType, TextEncoder, Tokenizer, WanPipelineConfig, WanPipelineError, WanPipelineOutput,
    WanT2VPipeline,
};
