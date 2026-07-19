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
    #[serde(rename = "_diffusers_version", default)]
    pub diffusers_version: Option<String>,
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
    /// Baked-in Diffusers configs for Wan 2.1 T2V 1.3B (consolidated / GGUF layout).
    pub fn wan21_t2v_13b() -> Self {
        Self {
            variant: WanVariant::Wan21T2v13B,
            model_index: WanModelIndex {
                class_name: "WanPipeline".to_string(),
                diffusers_version: Some("0.33.0.dev0".to_string()),
                scheduler: ("scheduler".into(), "UniPCMultistepScheduler".into()),
                text_encoder: ("text_encoder".into(), "UMT5EncoderModel".into()),
                tokenizer: ("tokenizer".into(), "T5Tokenizer".into()),
                transformer: ("transformer".into(), "WanTransformer3DModel".into()),
                vae: ("vae".into(), "AutoencoderKLWan".into()),
            },
            transformer: WanTransformerConfig {
                attention_head_dim: 128,
                cross_attn_norm: true,
                eps: 1e-6,
                ffn_dim: 8960,
                freq_dim: 256,
                in_channels: 16,
                num_attention_heads: 12,
                num_layers: 30,
                out_channels: 16,
                patch_size: [1, 2, 2],
                qk_norm: "rms_norm_across_heads".to_string(),
                rope_max_seq_len: 1024,
                text_dim: 4096,
                image_dim: None,
                added_kv_proj_dim: None,
            },
            vae: WanVaeConfig {
                base_dim: 96,
                dim_mult: vec![1, 2, 4, 4],
                dropout: 0.0,
                latents_mean: vec![
                    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                    -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
                ],
                latents_std: vec![
                    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                    2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
                ],
                num_res_blocks: 2,
                temporal_downsample: vec![false, true, true],
                z_dim: 16,
                attn_scales: vec![],
            },
            scheduler: WanSchedulerConfig {
                num_train_timesteps: 1000,
                flow_shift: 3.0,
                prediction_type: "flow_prediction".to_string(),
                use_flow_sigmas: true,
                solver_order: 2,
                solver_type: "bh2".to_string(),
                beta_start: 0.0001,
                beta_end: 0.02,
                beta_schedule: "linear".to_string(),
                timestep_spacing: "linspace".to_string(),
                predict_x0: true,
                lower_order_final: true,
                final_sigmas_type: "zero".to_string(),
                disable_corrector: vec![],
            },
            text_encoder: WanTextEncoderConfig {
                d_model: 4096,
                d_ff: 10240,
                d_kv: 64,
                num_heads: 64,
                num_layers: 24,
                vocab_size: 256_384,
                model_type: "umt5".to_string(),
                feed_forward_proj: "gated-gelu".to_string(),
                layer_norm_epsilon: 1e-6,
                dropout_rate: 0.1,
            },
        }
    }

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
pub fn infer_variant(transformer: &WanTransformerConfig) -> Result<WanVariant, String> {
    match (transformer.num_layers, transformer.num_attention_heads) {
        (30, 12) => Ok(WanVariant::Wan21T2v13B),
        (40, 40) => Ok(WanVariant::Wan21T2v14B),
        (layers, heads) => Err(format!(
            "unsupported Wan transformer architecture: expected known (layers, heads), got ({layers}, {heads})"
        )),
    }
}

/// Strictly validate the architecture represented by the local Wan 2.1 T2V 1.3B checkpoint.
///
/// The checkpoint was authored by Diffusers `0.33.0.dev0`; accepting a partially matching
/// configuration would make a wrong tensor layout fail later with an opaque shape error.
pub fn validate_wan21_t2v_13b(config: &WanFullConfig) -> Result<(), String> {
    macro_rules! check {
        ($field:expr, $expected:expr, $label:literal) => {
            if $field != $expected {
                return Err(format!(
                    "Wan2.1-T2V-1.3B config mismatch for {}: expected {:?}, got {:?}",
                    $label, $expected, $field
                ));
            }
        };
    }

    check!(
        config.model_index.class_name,
        "WanPipeline",
        "model_index._class_name"
    );
    check!(
        config.transformer.attention_head_dim,
        128usize,
        "transformer.attention_head_dim"
    );
    check!(
        config.transformer.cross_attn_norm,
        true,
        "transformer.cross_attn_norm"
    );
    check!(config.transformer.ffn_dim, 8960usize, "transformer.ffn_dim");
    check!(
        config.transformer.freq_dim,
        256usize,
        "transformer.freq_dim"
    );
    check!(
        config.transformer.in_channels,
        16usize,
        "transformer.in_channels"
    );
    check!(
        config.transformer.num_attention_heads,
        12usize,
        "transformer.num_attention_heads"
    );
    check!(
        config.transformer.num_layers,
        30usize,
        "transformer.num_layers"
    );
    check!(
        config.transformer.out_channels,
        16usize,
        "transformer.out_channels"
    );
    check!(
        config.transformer.patch_size,
        [1usize, 2, 2],
        "transformer.patch_size"
    );
    check!(
        config.transformer.qk_norm,
        "rms_norm_across_heads",
        "transformer.qk_norm"
    );
    check!(
        config.transformer.rope_max_seq_len,
        1024usize,
        "transformer.rope_max_seq_len"
    );
    check!(
        config.transformer.text_dim,
        4096usize,
        "transformer.text_dim"
    );
    check!(
        config.transformer.image_dim,
        None::<usize>,
        "transformer.image_dim"
    );
    check!(
        config.transformer.added_kv_proj_dim,
        None::<usize>,
        "transformer.added_kv_proj_dim"
    );
    if (config.transformer.eps - 1e-6).abs() > 1e-12 {
        return Err(format!(
            "Wan2.1-T2V-1.3B config mismatch for transformer.eps: expected {:?}, got {:?}",
            1e-6f64, config.transformer.eps
        ));
    }

    check!(config.vae.base_dim, 96usize, "vae.base_dim");
    check!(config.vae.dim_mult, vec![1usize, 2, 4, 4], "vae.dim_mult");
    check!(config.vae.num_res_blocks, 2usize, "vae.num_res_blocks");
    check!(
        config.vae.temporal_downsample,
        vec![false, true, true],
        "vae.temperal_downsample"
    );
    check!(config.vae.z_dim, 16usize, "vae.z_dim");
    if config.vae.latents_mean.len() != 16 || config.vae.latents_std.len() != 16 {
        return Err(format!(
            "Wan2.1-T2V-1.3B config mismatch for VAE latent statistics: expected 16 values, got mean={} std={}",
            config.vae.latents_mean.len(),
            config.vae.latents_std.len()
        ));
    }
    if config.vae.latents_std.iter().any(|value| *value <= 0.0) {
        return Err("Wan2.1-T2V-1.3B VAE latents_std must be positive".to_string());
    }

    check!(
        config.scheduler.num_train_timesteps,
        1000usize,
        "scheduler.num_train_timesteps"
    );
    check!(
        config.scheduler.prediction_type,
        "flow_prediction",
        "scheduler.prediction_type"
    );
    check!(
        config.scheduler.use_flow_sigmas,
        true,
        "scheduler.use_flow_sigmas"
    );
    check!(
        config.scheduler.solver_order,
        2usize,
        "scheduler.solver_order"
    );
    check!(config.scheduler.solver_type, "bh2", "scheduler.solver_type");
    check!(
        config.scheduler.timestep_spacing,
        "linspace",
        "scheduler.timestep_spacing"
    );

    check!(
        config.text_encoder.d_model,
        4096usize,
        "text_encoder.d_model"
    );
    check!(config.text_encoder.d_ff, 10240usize, "text_encoder.d_ff");
    check!(config.text_encoder.d_kv, 64usize, "text_encoder.d_kv");
    check!(
        config.text_encoder.num_heads,
        64usize,
        "text_encoder.num_heads"
    );
    check!(
        config.text_encoder.num_layers,
        24usize,
        "text_encoder.num_layers"
    );
    check!(
        config.text_encoder.vocab_size,
        256384usize,
        "text_encoder.vocab_size"
    );
    check!(
        config.text_encoder.model_type,
        "umt5",
        "text_encoder.model_type"
    );
    check!(
        config.text_encoder.feed_forward_proj,
        "gated-gelu",
        "text_encoder.feed_forward_proj"
    );
    Ok(())
}

/// Validate that a parsed index matches expected Wan Diffusers components.
pub fn validate_model_index(index: &WanModelIndex) -> Result<(), String> {
    if index.class_name != "WanPipeline" {
        return Err(format!("expected WanPipeline, got {}", index.class_name));
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
    if !matches!(
        index.scheduler.1.as_str(),
        "UniPCMultistepScheduler" | "FlowMatchEulerDiscreteScheduler"
    ) {
        return Err(format!("unexpected scheduler class: {}", index.scheduler.1));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_validation_reports_expected_and_actual_patch_size() {
        let mut config = WanFullConfig::wan21_t2v_13b();
        config.transformer.patch_size = [2, 2, 2];

        let error = validate_wan21_t2v_13b(&config).expect_err("mismatch must fail");

        assert!(error.contains("transformer.patch_size"));
        assert!(error.contains("expected"));
        assert!(error.contains("got"));
    }

    #[test]
    fn unknown_transformer_architecture_is_not_silently_downgraded() {
        let mut config = WanFullConfig::wan21_t2v_13b();
        config.transformer.num_layers = 31;

        let error = infer_variant(&config.transformer).expect_err("unknown architecture");

        assert!(error.contains("unsupported Wan transformer architecture"));
    }
}
