//! T5/UMT5 Text Encoder Configuration and Wrappers.
//!
//! Provides unified configuration for T5 family encoders:
//! - T5-XXL (used in LTX-Video)
//! - UMT5-XXL (used in Wan)
//!
//! Architecturally identical, only vocab_size differs.

use candle_nn::Activation;
use candle_transformers::models::t5;
use serde::{Deserialize, Serialize};
use std::path::Path;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for T5/UMT5 Text Encoders.
///
/// Works for both T5 (LTX-Video) and UMT5 (Wan) since architecture is identical.
/// Main difference is vocab_size: T5=32128, UMT5=250112.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5EncoderConfig {
    /// Model dimension (T5-XXL: 4096)
    pub d_model: usize,

    /// Feed-forward dimension (T5-XXL: 10240)
    #[serde(default = "default_d_ff")]
    pub d_ff: usize,

    /// Key/value dimension per head (T5-XXL: 64)
    #[serde(default = "default_d_kv")]
    pub d_kv: usize,

    /// Number of attention heads (T5-XXL: 64)
    pub num_heads: usize,

    /// Number of encoder layers (T5-XXL: 24)
    pub num_layers: usize,

    /// Vocabulary size (T5: 32128, UMT5: 250112)
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Layer norm epsilon
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,

    /// Relative attention number of buckets
    #[serde(default = "default_relative_attention_num_buckets")]
    pub relative_attention_num_buckets: usize,

    /// Relative attention max distance
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,

    /// Maximum sequence length for prompts
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Whether to use CPU offloading for memory efficiency
    #[serde(default)]
    pub cpu_offload: bool,

    /// Dropout rate (0.0 for inference)
    #[serde(default)]
    pub dropout_rate: f64,

    /// Feed-forward projection type: "gated-gelu" or "relu"
    #[serde(default = "default_feed_forward_proj")]
    pub feed_forward_proj: String,
}

// Default value functions for serde
fn default_d_ff() -> usize {
    10240
}
fn default_d_kv() -> usize {
    64
}
fn default_vocab_size() -> usize {
    32128
}
fn default_layer_norm_epsilon() -> f64 {
    1e-6
}
fn default_relative_attention_num_buckets() -> usize {
    32
}
fn default_relative_attention_max_distance() -> usize {
    128
}
fn default_max_seq_len() -> usize {
    256
}
fn default_feed_forward_proj() -> String {
    "gated-gelu".to_string()
}

impl Default for T5EncoderConfig {
    fn default() -> Self {
        Self::t5_xxl()
    }
}

impl T5EncoderConfig {
    /// Create a new T5 encoder configuration with custom dimensions.
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize) -> Self {
        Self {
            d_model,
            d_ff: d_model * 4,
            d_kv: d_model / num_heads,
            num_heads,
            num_layers,
            vocab_size: 32128,
            layer_norm_epsilon: 1e-6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            max_seq_len: 256,
            cpu_offload: false,
            dropout_rate: 0.0,
            feed_forward_proj: "gated-gelu".to_string(),
        }
    }

    // =========================================================================
    // T5 Presets (for LTX-Video)
    // =========================================================================

    /// T5-XXL configuration preset.
    ///
    /// Used in LTX-Video:
    /// - 4096 embedding dimension
    /// - 64 attention heads
    /// - 24 encoder layers
    /// - vocab_size: 32128
    pub fn t5_xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            vocab_size: 32128,
            layer_norm_epsilon: 1e-6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            max_seq_len: 256,
            cpu_offload: false,
            dropout_rate: 0.0,
            feed_forward_proj: "gated-gelu".to_string(),
        }
    }

    /// T5-Large configuration preset (for testing).
    pub fn t5_large() -> Self {
        Self {
            d_model: 1024,
            d_ff: 4096,
            d_kv: 64,
            num_heads: 16,
            num_layers: 24,
            vocab_size: 32128,
            layer_norm_epsilon: 1e-6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            max_seq_len: 256,
            cpu_offload: false,
            dropout_rate: 0.0,
            feed_forward_proj: "gated-gelu".to_string(),
        }
    }

    // =========================================================================
    // UMT5 Presets (for Wan)
    // =========================================================================

    /// UMT5-XXL configuration preset.
    ///
    /// Used in Wan T2V models:
    /// - 4096 embedding dimension
    /// - 64 attention heads
    /// - 24 encoder layers
    /// - vocab_size: 250112 (main difference from T5)
    /// - max_seq_len: 512 (Wan uses longer sequences)
    pub fn umt5_xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            vocab_size: 250112,  // UMT5 vocab size
            layer_norm_epsilon: 1e-6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            max_seq_len: 512,    // Wan uses 512
            cpu_offload: false,
            dropout_rate: 0.0,
            feed_forward_proj: "gated-gelu".to_string(),
        }
    }

    /// UMT5-Large preset (for testing).
    pub fn umt5_large() -> Self {
        Self {
            d_model: 1024,
            d_ff: 4096,
            d_kv: 64,
            num_heads: 16,
            num_layers: 24,
            vocab_size: 250112,
            layer_norm_epsilon: 1e-6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            max_seq_len: 512,
            cpu_offload: false,
            dropout_rate: 0.0,
            feed_forward_proj: "gated-gelu".to_string(),
        }
    }

    // =========================================================================
    // Configuration Methods
    // =========================================================================

    /// Load configuration from a JSON file.
    pub fn from_json(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })
    }

    /// Enable/disable CPU offloading.
    pub fn with_cpu_offload(mut self, enable: bool) -> Self {
        self.cpu_offload = enable;
        self
    }

    /// Set maximum sequence length.
    pub fn with_max_seq_len(mut self, max_len: usize) -> Self {
        self.max_seq_len = max_len;
        self
    }

    /// Set vocabulary size.
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Convert to candle-transformers T5 Config.
    pub fn to_candle_t5_config(&self) -> t5::Config {
        let (gated, activation) = match self.feed_forward_proj.as_str() {
            "gated-gelu" => (true, Activation::NewGelu),
            "gated-silu" => (true, Activation::Silu),
            "relu" => (false, Activation::Relu),
            _ => (true, Activation::NewGelu),
        };

        t5::Config {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            d_kv: self.d_kv,
            d_ff: self.d_ff,
            num_layers: self.num_layers,
            num_decoder_layers: None,
            num_heads: self.num_heads,
            relative_attention_num_buckets: self.relative_attention_num_buckets,
            relative_attention_max_distance: self.relative_attention_max_distance,
            dropout_rate: self.dropout_rate,
            layer_norm_epsilon: self.layer_norm_epsilon,
            initializer_factor: 1.0,
            feed_forward_proj: t5::ActivationWithOptionalGating { gated, activation },
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: false,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: None,
        }
    }

    /// Check if this is a UMT5 configuration (based on vocab_size).
    pub fn is_umt5(&self) -> bool {
        self.vocab_size > 100000
    }
}

// =============================================================================
// Embedding Cache
// =============================================================================

use std::collections::HashMap;
use candle_core::Tensor;

/// Cache for storing computed text embeddings.
#[derive(Debug)]
pub struct EmbeddingCache {
    cache: HashMap<String, Tensor>,
    hits: usize,
    enabled: bool,
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            enabled: false,
        }
    }

    pub fn enable(&mut self, enable: bool) {
        self.enabled = enable;
    }

    pub fn get(&mut self, key: &str) -> Option<&Tensor> {
        if self.enabled {
            if let Some(tensor) = self.cache.get(key) {
                self.hits += 1;
                Some(tensor)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: String, tensor: Tensor) {
        if self.enabled {
            self.cache.insert(key, tensor);
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.enabled && self.cache.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn hits(&self) -> usize {
        self.hits
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_xxl_config() {
        let config = T5EncoderConfig::t5_xxl();
        assert_eq!(config.vocab_size, 32128);
        assert_eq!(config.d_model, 4096);
        assert!(!config.is_umt5());
    }

    #[test]
    fn test_umt5_xxl_config() {
        let config = T5EncoderConfig::umt5_xxl();
        assert_eq!(config.vocab_size, 250112);
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.max_seq_len, 512);
        assert!(config.is_umt5());
    }

    #[test]
    fn test_candle_config_conversion() {
        let config = T5EncoderConfig::t5_xxl();
        let candle_config = config.to_candle_t5_config();
        assert_eq!(candle_config.vocab_size, 32128);
        assert_eq!(candle_config.d_model, 4096);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new();
        cache.enable(true);
        
        assert!(!cache.contains("test"));
        assert_eq!(cache.len(), 0);
    }
}
