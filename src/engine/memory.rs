//! Memory and precision options shared across video generation backends.

use serde::{Deserialize, Serialize};

/// Controls how models use device memory during generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryOptions {
    pub vae_tiling: bool,
    pub vae_slicing: bool,
    pub cpu_offload: bool,
}

/// Precision preferences for model components.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionOptions {
    Fp32,
    #[default]
    Bf16,
    Fp16,
}
