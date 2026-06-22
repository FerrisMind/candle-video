//! Model registry, capabilities, and load specifications.

use std::path::PathBuf;

use candle_core::DType;
use serde::{Deserialize, Serialize};

use super::memory::{MemoryOptions, PrecisionOptions};
use super::video::{FrameRule, Resolution, VideoTask};

/// Identifies a supported video generation model variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoModelSpec {
    Ltx {
        version: String,
        weights: PathBuf,
        model_id: Option<String>,
    },
    Wan21T2v13b {
        weights: PathBuf,
        dtype: DType,
    },
    Wan21T2v14b {
        weights: PathBuf,
        dtype: DType,
    },
    Wan22T2vA14b {
        weights: PathBuf,
        dtype: DType,
    },
    Wan22I2vA14b {
        weights: PathBuf,
        dtype: DType,
    },
    Wan22Ti2v5b {
        weights: PathBuf,
        dtype: DType,
    },
}

impl VideoModelSpec {
    pub fn model_id_str(&self) -> &'static str {
        match self {
            Self::Ltx { .. } => "ltx",
            Self::Wan21T2v13b { .. } => "wan:2.1-t2v-1.3b",
            Self::Wan21T2v14b { .. } => "wan:2.1-t2v-14b",
            Self::Wan22T2vA14b { .. } => "wan:2.2-t2v-a14b",
            Self::Wan22I2vA14b { .. } => "wan:2.2-i2v-a14b",
            Self::Wan22Ti2v5b { .. } => "wan:2.2-ti2v-5b",
        }
    }
}

/// Describes what a loaded backend can do.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub model_id: String,
    pub tasks: Vec<VideoTask>,
    pub resolutions: Vec<Resolution>,
    pub frame_rule: FrameRule,
    pub supports_negative_prompt: bool,
    pub supports_lora: bool,
    pub supports_quantized_text_encoder: bool,
    pub supports_two_stage_denoising: bool,
}

impl ModelCapabilities {
    pub fn ltx(version: &str) -> Self {
        Self {
            model_id: format!("ltx:{version}"),
            tasks: vec![VideoTask::TextToVideo],
            resolutions: vec![
                Resolution {
                    width: 768,
                    height: 512,
                },
                Resolution {
                    width: 1280,
                    height: 704,
                },
            ],
            frame_rule: FrameRule {
                min_frames: 9,
                max_frames: 257,
                step: Some(8),
            },
            supports_negative_prompt: true,
            supports_lora: false,
            supports_quantized_text_encoder: true,
            supports_two_stage_denoising: false,
        }
    }

    pub fn wan21_t2v_13b() -> Self {
        Self {
            model_id: "wan:2.1-t2v-1.3b".to_string(),
            tasks: vec![VideoTask::TextToVideo],
            resolutions: vec![Resolution {
                width: 832,
                height: 480,
            }],
            frame_rule: FrameRule {
                min_frames: 5,
                max_frames: 121,
                step: Some(4),
            },
            supports_negative_prompt: true,
            supports_lora: false,
            supports_quantized_text_encoder: true,
            supports_two_stage_denoising: false,
        }
    }
}

/// Static registry of known model identifiers.
#[derive(Debug, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    pub fn parse(model: &str) -> Option<VideoModelSpec> {
        match model {
            "wan:2.1-t2v-1.3b" | "wan2.1-t2v-1.3b" => None, // needs weights path
            "ltx:0.9.8-2b-distilled" => None,
            _ => None,
        }
    }

    pub fn known_models() -> &'static [&'static str] {
        &[
            "ltx:0.9.8-2b-distilled",
            "wan:2.1-t2v-1.3b",
            "wan:2.2-ti2v-5b",
        ]
    }
}

/// Load-time options bundled with a model spec.
#[derive(Debug, Clone, Default)]
pub struct EngineLoadOptions {
    pub precision: PrecisionOptions,
    pub memory: MemoryOptions,
}
