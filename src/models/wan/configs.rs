//! Wan 2.1 / 2.2 Diffusers configuration types.

use serde::{Deserialize, Serialize};

/// Supported Wan model variants (extensible for future backends).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WanVariant {
    Wan21T2v13B,
    Wan21T2v14B,
    Wan22T2vA14B,
    Wan22I2vA14B,
    Wan22Ti2v5B,
}

impl WanVariant {
    pub fn is_implemented(&self) -> bool {
        matches!(self, Self::Wan21T2v13B)
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Wan21T2v13B => "Wan2.1-T2V-1.3B",
            Self::Wan21T2v14B => "Wan2.1-T2V-14B",
            Self::Wan22T2vA14B => "Wan2.2-T2V-A14B",
            Self::Wan22I2vA14B => "Wan2.2-I2V-A14B",
            Self::Wan22Ti2v5B => "Wan2.2-TI2V-5B",
        }
    }
}

/// Parsed `model_index.json` for a Diffusers Wan checkpoint folder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanModelIndex {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub scheduler: (String, String),
    pub text_encoder: (String, String),
    pub tokenizer: (String, String),
    pub transformer: (String, String),
    pub vae: (String, String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanTransformerConfig {
    pub attention_head_dim: usize,
    pub cross_attn_norm: bool,
    pub eps: f64,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub in_channels: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub out_channels: usize,
    pub patch_size: [usize; 3],
    pub qk_norm: String,
    pub rope_max_seq_len: usize,
    pub text_dim: usize,
    #[serde(default)]
    pub image_dim: Option<usize>,
    #[serde(default)]
    pub added_kv_proj_dim: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanVaeConfig {
    pub base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub dropout: f32,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub num_res_blocks: usize,
    #[serde(rename = "temperal_downsample")]
    pub temporal_downsample: Vec<bool>,
    pub z_dim: usize,
    #[serde(default)]
    pub attn_scales: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanSchedulerConfig {
    pub num_train_timesteps: usize,
    pub flow_shift: f32,
    pub prediction_type: String,
    pub use_flow_sigmas: bool,
    pub solver_order: usize,
    pub solver_type: String,
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: String,
    pub timestep_spacing: String,
    #[serde(default = "default_true")]
    pub predict_x0: bool,
    #[serde(default = "default_true")]
    pub lower_order_final: bool,
    #[serde(default = "default_final_sigmas_zero")]
    pub final_sigmas_type: String,
    #[serde(default)]
    pub disable_corrector: Vec<usize>,
}

fn default_true() -> bool {
    true
}

fn default_final_sigmas_zero() -> String {
    "zero".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanTextEncoderConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub model_type: String,
    pub feed_forward_proj: String,
    pub layer_norm_epsilon: f64,
    pub dropout_rate: f64,
}

/// Full component configs discovered from a Wan Diffusers directory.
#[derive(Debug, Clone)]
pub struct WanFullConfig {
    pub variant: WanVariant,
    pub model_index: WanModelIndex,
    pub transformer: WanTransformerConfig,
    pub vae: WanVaeConfig,
    pub scheduler: WanSchedulerConfig,
    pub text_encoder: WanTextEncoderConfig,
}

impl WanFullConfig {
    pub fn spatial_compression(&self) -> usize {
        // patch (1,2,2) * VAE spatial downsampling (8x for Wan 2.1 T2V 1.3B)
        8
    }

    pub fn temporal_compression(&self) -> usize {
        4
    }

    pub fn latent_channels(&self) -> usize {
        self.transformer.in_channels
    }
}

/// Infer variant from transformer config heuristics.
pub fn infer_variant(transformer: &WanTransformerConfig) -> WanVariant {
    match (transformer.num_layers, transformer.num_attention_heads) {
        (30, 12) => WanVariant::Wan21T2v13B,
        (40, 40) => WanVariant::Wan21T2v14B,
        _ => WanVariant::Wan21T2v13B,
    }
}

/// Validate that a parsed index matches expected Wan Diffusers components.
pub fn validate_model_index(index: &WanModelIndex) -> Result<(), String> {
    if index.class_name != "WanPipeline" {
        return Err(format!(
            "expected WanPipeline, got {}",
            index.class_name
        ));
    }
    if index.transformer.1 != "WanTransformer3DModel" {
        return Err(format!(
            "unexpected transformer class: {}",
            index.transformer.1
        ));
    }
    if index.vae.1 != "AutoencoderKLWan" {
        return Err(format!("unexpected VAE class: {}", index.vae.1));
    }
    if index.text_encoder.1 != "UMT5EncoderModel" {
        return Err(format!(
            "unexpected text encoder class: {}",
            index.text_encoder.1
        ));
    }
    Ok(())
}
