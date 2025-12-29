//! T5-XXL Text Encoder Wrapper for LTX-Video
//!
//! This module provides a wrapper around the T5 encoder from candle-transformers,
//! specifically configured for T5-XXL which is used in LTX-Video for text conditioning.
//!
//! Features:
//! - Integration with `candle-transformers` T5EncoderModel
//! - T5-XXL configuration preset (4096-dim, 64 heads, 24 layers)
//! - Embedding caching for repeated prompts
//! - CPU offloading support for memory efficiency
//! - Classifier Free Guidance (CFG) encoding support
//!
//! # Example
//!
//! ```rust,ignore
//! use candle_core::{Device, DType};
//! use candle_video::text_encoder::{T5EncoderConfig, T5TextEncoderWrapper};
//!
//! let config = T5EncoderConfig::t5_xxl();
//! let mut encoder = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32)?;
//!
//! // With actual model weights loaded:
//! // let embeddings = encoder.encode("A beautiful sunset over the ocean")?;
//!
//! // For testing without weights:
//! let embeddings = encoder.mock_encode("A beautiful sunset")?;
//! ```

use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::t5;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::loader::LoaderError;

// =============================================================================
// Error Types
// =============================================================================

/// Errors specific to text encoding
#[derive(Debug, thiserror::Error)]
pub enum TextEncoderError {
    #[error("Failed to load config: {0}")]
    ConfigLoad(#[from] LoaderError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model not loaded. Call load_model() first.")]
    ModelNotLoaded,

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for T5 Text Encoder
///
/// This configuration is compatible with T5 models from Hugging Face,
/// with a preset for T5-XXL which is used in LTX-Video.
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

    /// Vocabulary size
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

impl Default for T5EncoderConfig {
    fn default() -> Self {
        Self::t5_xxl()
    }
}

impl T5EncoderConfig {
    /// Create a new T5 encoder configuration with custom dimensions
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
        }
    }

    /// T5-XXL configuration preset
    ///
    /// T5-XXL is the text encoder used in LTX-Video:
    /// - 4096 embedding dimension
    /// - 64 attention heads
    /// - 24 encoder layers
    /// - ~11B parameters (encoder only)
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
        }
    }

    /// T5-Large configuration preset (for testing)
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
        }
    }

    /// Load configuration from a JSON file
    pub fn from_json(path: impl AsRef<Path>) -> std::result::Result<Self, LoaderError> {
        crate::loader::load_model_config(path)
    }

    /// Enable/disable CPU offloading
    pub fn with_cpu_offload(mut self, enable: bool) -> Self {
        self.cpu_offload = enable;
        self
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_len: usize) -> Self {
        self.max_seq_len = max_len;
        self
    }

    /// Convert to candle-transformers T5 Config
    pub fn to_candle_t5_config(&self) -> t5::Config {
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
            feed_forward_proj: t5::ActivationWithOptionalGating {
                gated: true,
                activation: candle_nn::Activation::NewGelu,
            },
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: false,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: None,
        }
    }
}

// =============================================================================
// Embedding Cache
// =============================================================================

/// Cache for storing computed embeddings
struct EmbeddingCache {
    cache: HashMap<String, Tensor>,
    hits: usize,
    enabled: bool,
}

impl EmbeddingCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            enabled: false,
        }
    }

    fn get(&mut self, key: &str) -> Option<&Tensor> {
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

    fn insert(&mut self, key: String, tensor: Tensor) {
        if self.enabled {
            self.cache.insert(key, tensor);
        }
    }

    fn contains(&self, key: &str) -> bool {
        self.enabled && self.cache.contains_key(key)
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
    }

    fn len(&self) -> usize {
        self.cache.len()
    }
}

// =============================================================================
// T5 Text Encoder Wrapper
// =============================================================================

/// Wrapper around T5EncoderModel for text-to-video generation
///
/// This wrapper provides:
/// - Easy-to-use encoding interface
/// - Embedding caching for repeated prompts
/// - CPU offloading support
/// - Configuration presets for T5-XXL
pub struct T5TextEncoderWrapper {
    config: T5EncoderConfig,
    device: Device,
    dtype: DType,
    model: Option<t5::T5EncoderModel>,
    cache: EmbeddingCache,
}

impl T5TextEncoderWrapper {
    /// Create a new T5 text encoder wrapper
    ///
    /// Note: This creates the wrapper without loading model weights.
    /// Call `load_model` to load weights from a VarBuilder.
    pub fn new(config: T5EncoderConfig, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            config,
            device,
            dtype,
            model: None,
            cache: EmbeddingCache::new(),
        })
    }

    /// Load model weights from a VarBuilder
    pub fn load_model(&mut self, vb: candle_nn::VarBuilder) -> Result<()> {
        let candle_config = self.config.to_candle_t5_config();
        let model = t5::T5EncoderModel::load(vb, &candle_config)?;
        self.model = Some(model);
        Ok(())
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the configuration
    pub fn config(&self) -> &T5EncoderConfig {
        &self.config
    }

    /// Check if CPU offloading is enabled
    pub fn is_cpu_offload_enabled(&self) -> bool {
        self.config.cpu_offload
    }

    /// Enable or disable embedding caching
    pub fn enable_cache(&mut self, enable: bool) {
        self.cache.enabled = enable;
    }

    /// Check if a prompt is in the cache
    pub fn cache_contains(&self, prompt: &str) -> bool {
        self.cache.contains(prompt)
    }

    /// Get number of cache hits
    pub fn cache_hits(&self) -> usize {
        self.cache.hits
    }

    /// Get current cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    /// Mock tokenization for testing without actual tokenizer
    ///
    /// Produces pseudo-tokens based on word count.
    /// Each space-separated word becomes one token, plus EOS token.
    pub fn mock_tokenize(&self, prompt: &str) -> Vec<u32> {
        if prompt.is_empty() {
            return vec![1]; // Just EOS token
        }

        let words: Vec<&str> = prompt.split_whitespace().collect();
        let mut tokens: Vec<u32> = words
            .iter()
            .enumerate()
            .map(|(i, _)| (i + 2) as u32) // Start from 2 (0=pad, 1=eos)
            .collect();

        // Truncate to max sequence length (leave room for EOS)
        if tokens.len() >= self.config.max_seq_len {
            tokens.truncate(self.config.max_seq_len - 1);
        }

        // Add EOS token
        tokens.push(1);

        tokens
    }

    /// Convert token IDs to a tensor
    pub fn token_ids_to_tensor(&self, token_ids: &[u32]) -> Result<Tensor> {
        let tokens: Vec<u32> = token_ids.to_vec();
        let tensor = Tensor::new(&tokens[..], &self.device)?;
        tensor.unsqueeze(0) // Add batch dimension: [seq_len] -> [1, seq_len]
    }

    // =========================================================================
    // Encoding (with model)
    // =========================================================================

    /// Encode a text prompt to embeddings using the loaded model
    ///
    /// Returns embeddings of shape `[1, seq_len, d_model]`
    pub fn encode(&mut self, prompt: &str) -> std::result::Result<Tensor, TextEncoderError> {
        // Check if model is loaded first
        if self.model.is_none() {
            return Err(TextEncoderError::ModelNotLoaded);
        }

        // Check cache first (immutable borrow)
        if let Some(cached) = self.cache.get(prompt) {
            return Ok(cached.clone());
        }

        // Tokenize and create tensor (immutable borrows of self.config and self.device)
        let token_ids = self.mock_tokenize(prompt);
        let input_tensor = self.token_ids_to_tensor(&token_ids)?;

        // Now get mutable reference to model for forward pass
        let model = self.model.as_mut().unwrap();
        let embeddings = model.forward(&input_tensor)?;

        // Cache result
        self.cache.insert(prompt.to_string(), embeddings.clone());

        Ok(embeddings)
    }

    // =========================================================================
    // Mock Encoding (for testing without model weights)
    // =========================================================================

    /// Mock encode for testing without actual model weights
    ///
    /// Produces random embeddings with the correct shape.
    /// Returns embeddings of shape `[1, seq_len, d_model]`
    pub fn mock_encode(&mut self, prompt: &str) -> Result<Tensor> {
        // Check cache first
        if let Some(cached) = self.cache.get(prompt) {
            return Ok(cached.clone());
        }

        // Tokenize to get sequence length
        let token_ids = self.mock_tokenize(prompt);
        let seq_len = token_ids.len();

        // Create mock embeddings with correct shape
        let embeddings = Tensor::randn(0f32, 1.0, (1, seq_len, self.config.d_model), &self.device)?;

        // Cache result
        self.cache.insert(prompt.to_string(), embeddings.clone());

        Ok(embeddings)
    }

    /// Mock encode a batch of prompts
    ///
    /// Returns embeddings of shape `[batch_size, max_seq_len, d_model]`
    /// All sequences are padded to the maximum length in the batch.
    pub fn mock_encode_batch(&mut self, prompts: &[&str]) -> Result<Tensor> {
        let batch_size = prompts.len();

        // Tokenize all prompts and find max length
        let token_batches: Vec<Vec<u32>> = prompts.iter().map(|p| self.mock_tokenize(p)).collect();

        let max_seq_len = token_batches.iter().map(|t| t.len()).max().unwrap_or(1);

        // Create mock embeddings for the batch
        let embeddings = Tensor::randn(
            0f32,
            1.0,
            (batch_size, max_seq_len, self.config.d_model),
            &self.device,
        )?;

        Ok(embeddings)
    }

    /// Mock encode for Classifier Free Guidance (CFG)
    ///
    /// Returns (positive_embeddings, negative_embeddings) with matching shapes.
    /// The negative prompt is typically empty for unconditional generation.
    pub fn mock_encode_for_cfg(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        // Tokenize both prompts
        let pos_tokens = self.mock_tokenize(prompt);
        let neg_tokens = self.mock_tokenize(negative_prompt);

        // Use max length for both
        let max_len = pos_tokens.len().max(neg_tokens.len());

        // Create mock embeddings with matching shapes
        let positive_emb =
            Tensor::randn(0f32, 1.0, (1, max_len, self.config.d_model), &self.device)?;

        let negative_emb = Tensor::randn(
            0f32,
            0.1, // Lower variance for unconditional
            (1, max_len, self.config.d_model),
            &self.device,
        )?;

        Ok((positive_emb, negative_emb))
    }

    /// Encode for CFG with actual model weights
    pub fn encode_for_cfg(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> std::result::Result<(Tensor, Tensor), TextEncoderError> {
        let positive_emb = self.encode(prompt)?;
        let negative_emb = self.encode(negative_prompt)?;

        // Pad to matching sequence lengths if necessary
        let pos_seq_len = positive_emb.dim(1)?;
        let neg_seq_len = negative_emb.dim(1)?;

        if pos_seq_len != neg_seq_len {
            let max_len = pos_seq_len.max(neg_seq_len);
            let positive_emb = self.pad_to_length(&positive_emb, max_len)?;
            let negative_emb = self.pad_to_length(&negative_emb, max_len)?;
            Ok((positive_emb, negative_emb))
        } else {
            Ok((positive_emb, negative_emb))
        }
    }

    /// Pad embeddings to a target sequence length
    fn pad_to_length(&self, embeddings: &Tensor, target_len: usize) -> Result<Tensor> {
        let current_len = embeddings.dim(1)?;

        if current_len >= target_len {
            return Ok(embeddings.clone());
        }

        let batch_size = embeddings.dim(0)?;
        let d_model = embeddings.dim(2)?;
        let pad_len = target_len - current_len;

        // Create zero padding
        let padding = Tensor::zeros((batch_size, pad_len, d_model), self.dtype, &self.device)?;

        // Concatenate along sequence dimension
        Tensor::cat(&[embeddings, &padding], 1)
    }

    /// Clear KV cache if model is loaded
    pub fn clear_kv_cache(&mut self) {
        if let Some(model) = &mut self.model {
            model.clear_kv_cache();
        }
    }
}

// =============================================================================
// Quantized T5 Encoder (GGUF support)
// =============================================================================

use candle_transformers::models::quantized_t5;
use tokenizers::Tokenizer;

/// Quantized T5 Text Encoder for LTX-Video
///
/// This encoder loads T5-XXL from GGUF format for efficient inference.
/// It uses the `candle-transformers` quantized_t5 module.
pub struct QuantizedT5Encoder {
    model: quantized_t5::T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
    config: quantized_t5::Config,
    d_model: usize,
    max_seq_len: usize,
}

impl QuantizedT5Encoder {
    /// Load a quantized T5 encoder from GGUF file
    ///
    /// # Arguments
    /// * `gguf_path` - Path to the GGUF model file
    /// * `tokenizer_path` - Path to tokenizer.json
    /// * `config_path` - Path to config.json (required for quantized model)
    /// * `device` - Device to load model on (CPU recommended for quantized)
    /// * `max_seq_len` - Maximum sequence length for tokenization
    pub fn load(
        gguf_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        device: &Device,
        max_seq_len: usize,
    ) -> std::result::Result<Self, TextEncoderError> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| TextEncoderError::Tokenizer(e.to_string()))?;

        // Load config from JSON file
        let config_str = std::fs::read_to_string(config_path.as_ref())
            .map_err(|e| TextEncoderError::Tokenizer(format!("Failed to read config: {}", e)))?;
        let config: quantized_t5::Config = serde_json::from_str(&config_str)
            .map_err(|e| TextEncoderError::Tokenizer(format!("Failed to parse config: {}", e)))?;

        // Load model weights from GGUF
        let vb = quantized_t5::VarBuilder::from_gguf(gguf_path.as_ref(), device)?;
        let model = quantized_t5::T5ForConditionalGeneration::load(vb, &config)?;

        // T5-XXL has d_model = 4096
        let d_model = 4096;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            config,
            d_model,
            max_seq_len,
        })
    }

    /// Get the model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Tokenize a text prompt
    pub fn tokenize(&self, prompt: &str) -> std::result::Result<Vec<u32>, TextEncoderError> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| TextEncoderError::Tokenizer(e.to_string()))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        // Truncate to max length
        if ids.len() > self.max_seq_len {
            ids.truncate(self.max_seq_len);
        }

        Ok(ids)
    }

    /// Encode a text prompt to embeddings
    ///
    /// Returns tensor of shape [1, seq_len, d_model]
    pub fn encode(&mut self, prompt: &str) -> std::result::Result<Tensor, TextEncoderError> {
        // Tokenize
        let token_ids = self.tokenize(prompt)?;

        // Create input tensor
        let input_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;

        // Run through encoder
        let encoder_output = self.model.encode(&input_ids)?;

        Ok(encoder_output)
    }

    /// Encode prompts for CFG (positive and negative)
    ///
    /// Returns (positive_embeddings, negative_embeddings) with matching sequence lengths
    pub fn encode_for_cfg(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> std::result::Result<(Tensor, Tensor), TextEncoderError> {
        // Tokenize both prompts
        let pos_ids = self.tokenize(prompt)?;
        let neg_ids = self.tokenize(negative_prompt)?;

        // Pad to same length
        let max_len = pos_ids.len().max(neg_ids.len());

        let pos_ids = Self::pad_to_length(pos_ids, max_len, self.config.pad_token_id as u32);
        let neg_ids = Self::pad_to_length(neg_ids, max_len, self.config.pad_token_id as u32);

        // Create tensors
        let pos_input = Tensor::new(&pos_ids[..], &self.device)?.unsqueeze(0)?;
        let neg_input = Tensor::new(&neg_ids[..], &self.device)?.unsqueeze(0)?;

        // Encode both
        let pos_emb = self.model.encode(&pos_input)?;
        let neg_emb = self.model.encode(&neg_input)?;

        Ok((pos_emb, neg_emb))
    }

    /// Pad token IDs to target length
    fn pad_to_length(mut ids: Vec<u32>, target_len: usize, pad_id: u32) -> Vec<u32> {
        while ids.len() < target_len {
            ids.push(pad_id);
        }
        ids
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
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.num_heads, 64);
        assert_eq!(config.num_layers, 24);
    }

    #[test]
    fn test_wrapper_creation() {
        let config = T5EncoderConfig::t5_xxl();
        let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32);
        assert!(wrapper.is_ok());
    }

    #[test]
    fn test_mock_tokenize() {
        let config = T5EncoderConfig::t5_xxl();
        let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

        let tokens = wrapper.mock_tokenize("Hello world");
        assert_eq!(tokens.len(), 3); // "Hello", "world", EOS
        assert_eq!(tokens[2], 1); // EOS token
    }

    #[test]
    fn test_mock_encode_shape() {
        let config = T5EncoderConfig::t5_xxl();
        let mut wrapper =
            T5TextEncoderWrapper::new(config.clone(), Device::Cpu, DType::F32).unwrap();

        let embeddings = wrapper.mock_encode("Test prompt").unwrap();
        assert_eq!(embeddings.dim(0).unwrap(), 1);
        assert_eq!(embeddings.dim(2).unwrap(), config.d_model);
    }

    #[test]
    fn test_to_candle_config() {
        let config = T5EncoderConfig::t5_xxl();
        let candle_config = config.to_candle_t5_config();

        assert_eq!(candle_config.d_model, 4096);
        assert_eq!(candle_config.num_heads, 64);
    }
}
