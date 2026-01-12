//! UMT5 Text Encoder for Wan T2V Pipeline.
//!
//! This module provides text encoder implementations for Wan models:
//! - `UMT5TextEncoder` - Full precision UMT5-XXL encoder using candle-transformers
//! - `QuantizedUMT5Encoder` - GGUF quantized UMT5 encoder for memory efficiency
//!
//! Both implement the `TextEncoder` trait from `t2v_pipeline_v2.rs`.
//!
//! # Architecture
//!
//! UMT5 is architecturally identical to T5, with the main difference being:
//! - vocab_size: 256384 (vs T5's 32128)
//! - max_seq_len: 512 (Wan uses longer sequences)
//!
//! # Example
//!
//! ```rust,ignore
//! use candle_core::{Device, DType};
//! use candle_video::models::wan::text_encoder::{UMT5TextEncoder, load_umt5_encoder};
//!
//! // Load from safetensors
//! let encoder = load_umt5_encoder(
//!     "models/wan/text_encoder",
//!     &Device::Cuda(0),
//!     DType::BF16,
//! )?;
//!
//! // Encode text
//! let embeddings = encoder.encode(&input_ids, &attention_mask)?;
//! // embeddings shape: [batch, seq_len, 4096]
//! ```

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5;
use std::path::Path;
use tokenizers::Tokenizer;

use super::t2v_pipeline_v2::{TextEncoder, Tokenizer as TokenizerTrait, TokenizerOutput};
use crate::interfaces::quantized_t5_encoder::{QuantizedT5Config, QuantizedT5EncoderModel};
use crate::interfaces::t5_encoder::T5EncoderConfig;

// =============================================================================
// UMT5 Text Encoder (Full Precision)
// =============================================================================

/// UMT5-XXL Text Encoder for Wan T2V models.
///
/// Uses candle-transformers T5EncoderModel with UMT5 configuration.
/// Architecturally identical to T5, just different vocab_size.
pub struct UMT5TextEncoder {
    model: t5::T5EncoderModel,
    config: T5EncoderConfig,
    device: Device,
    dtype: DType,
}

impl UMT5TextEncoder {
    /// Create a new UMT5 encoder from VarBuilder.
    ///
    /// # Arguments
    /// * `vb` - VarBuilder with loaded weights
    /// * `config` - UMT5 configuration (use `T5EncoderConfig::umt5_xxl()`)
    pub fn new(vb: VarBuilder, config: T5EncoderConfig) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let candle_config = config.to_candle_t5_config();
        let model = t5::T5EncoderModel::load(vb, &candle_config)?;

        Ok(Self {
            model,
            config,
            device,
            dtype,
        })
    }

    /// Create with default UMT5-XXL configuration.
    pub fn new_umt5_xxl(vb: VarBuilder) -> Result<Self> {
        Self::new(vb, T5EncoderConfig::umt5_xxl())
    }

    /// Get the model dimension (4096 for UMT5-XXL).
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Get vocabulary size (256384 for UMT5).
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forward pass through encoder.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    ///
    /// # Returns
    /// Hidden states [batch, seq_len, d_model]
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids)
    }

    /// Clear KV cache.
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

impl TextEncoder for UMT5TextEncoder {
    fn encode(&mut self, input_ids: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // T5EncoderModel doesn't use attention_mask in forward,
        // but we keep the interface for compatibility
        self.forward(input_ids)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

// =============================================================================
// Quantized UMT5 Encoder (GGUF)
// =============================================================================

/// Quantized UMT5 Text Encoder for memory-efficient inference.
///
/// Loads from GGUF format using the shared quantized T5 implementation.
pub struct QuantizedUMT5Encoder {
    model: QuantizedT5EncoderModel,
    dtype: DType,
}

impl QuantizedUMT5Encoder {
    /// Load quantized UMT5 from GGUF file.
    ///
    /// # Arguments
    /// * `gguf_path` - Path to GGUF model file
    /// * `device` - Device to load on (CPU recommended for quantized)
    pub fn load(gguf_path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let model = QuantizedT5EncoderModel::load_umt5(gguf_path, device)?;
        Ok(Self {
            model,
            dtype: DType::F32, // Quantized models dequantize to F32
        })
    }

    /// Load with custom config.
    pub fn load_with_config(
        gguf_path: impl AsRef<Path>,
        device: &Device,
        config: QuantizedT5Config,
    ) -> Result<Self> {
        let model = QuantizedT5EncoderModel::load_with_config(gguf_path, device, config)?;
        Ok(Self {
            model,
            dtype: DType::F32,
        })
    }

    /// Get model dimension.
    pub fn d_model(&self) -> usize {
        self.model.d_model()
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Check if this is a UMT5 model.
    pub fn is_umt5(&self) -> bool {
        self.model.is_umt5()
    }
}

impl TextEncoder for QuantizedUMT5Encoder {
    fn encode(&mut self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, Some(attention_mask))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

// =============================================================================
// UMT5 Tokenizer Wrapper
// =============================================================================

/// Tokenizer wrapper for UMT5 that implements the Tokenizer trait.
///
/// Uses HuggingFace tokenizers library.
pub struct UMT5Tokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
    pad_token_id: u32,
}

impl UMT5Tokenizer {
    /// Load tokenizer from file.
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json
    /// * `max_length` - Maximum sequence length (512 for Wan)
    pub fn load(path: impl AsRef<Path>, max_length: usize) -> std::result::Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // UMT5 uses pad_token_id = 0
        Ok(Self {
            tokenizer,
            max_length,
            pad_token_id: 0,
        })
    }

    /// Create from existing tokenizer.
    pub fn from_tokenizer(tokenizer: Tokenizer, max_length: usize) -> Self {
        Self {
            tokenizer,
            max_length,
            pad_token_id: 0,
        }
    }

    /// Get the underlying tokenizer.
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Tokenize a single text.
    pub fn tokenize(&self, text: &str) -> std::result::Result<Vec<u32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization error: {}", e))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        // Truncate if needed
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }

        Ok(ids)
    }
}

impl TokenizerTrait for UMT5Tokenizer {
    fn encode(
        &self,
        texts: &[String],
        max_len: usize,
    ) -> std::result::Result<TokenizerOutput, super::t2v_pipeline_v2::WanPipelineError> {
        let max_len = max_len.min(self.max_length);

        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        let mut all_masks: Vec<Vec<u8>> = Vec::with_capacity(texts.len());

        for text in texts {
            let encoding = self.tokenizer.encode(text.as_str(), true).map_err(|e| {
                super::t2v_pipeline_v2::WanPipelineError::InvalidArgument(format!(
                    "Tokenization error: {}",
                    e
                ))
            })?;

            let mut ids: Vec<u32> = encoding.get_ids().to_vec();
            let original_len = ids.len().min(max_len);

            // Truncate
            ids.truncate(max_len);

            // Create attention mask (1 for real tokens, 0 for padding)
            let mut mask = vec![1u8; original_len];

            // Pad to max_len
            while ids.len() < max_len {
                ids.push(self.pad_token_id);
                mask.push(0);
            }

            all_ids.push(ids);
            all_masks.push(mask);
        }

        Ok((all_ids, all_masks))
    }
}

// =============================================================================
// Loading Functions
// =============================================================================

/// Load UMT5 encoder from safetensors directory.
///
/// Expects directory structure:
/// ```text
/// text_encoder/
///   model.safetensors (or diffusion_pytorch_model.safetensors)
/// ```
///
/// # Arguments
/// * `dir` - Directory containing model weights
/// * `device` - Device to load on
/// * `dtype` - Data type (BF16 recommended)
pub fn load_umt5_encoder(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<UMT5TextEncoder> {
    let dir = dir.as_ref();

    // Try common file names
    let candidates = [
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
    ];

    for candidate in &candidates {
        let path = dir.join(candidate);
        if path.exists() {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], dtype, device)? };
            return UMT5TextEncoder::new_umt5_xxl(vb);
        }
    }

    candle_core::bail!("No model file found in {}", dir.display())
}

/// Load UMT5 encoder from a single safetensors file.
pub fn load_umt5_encoder_from_file(
    path: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<UMT5TextEncoder> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
    UMT5TextEncoder::new_umt5_xxl(vb)
}

/// Load quantized UMT5 encoder from GGUF file.
pub fn load_quantized_umt5_encoder(
    gguf_path: impl AsRef<Path>,
    device: &Device,
) -> Result<QuantizedUMT5Encoder> {
    QuantizedUMT5Encoder::load(gguf_path, device)
}

/// Load UMT5 tokenizer.
///
/// # Arguments
/// * `path` - Path to tokenizer.json
/// * `max_length` - Maximum sequence length (default 512 for Wan)
pub fn load_umt5_tokenizer(
    path: impl AsRef<Path>,
    max_length: Option<usize>,
) -> std::result::Result<UMT5Tokenizer, String> {
    UMT5Tokenizer::load(path, max_length.unwrap_or(512))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_umt5_config() {
        let config = T5EncoderConfig::umt5_xxl();
        assert_eq!(config.vocab_size, 256384);
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.max_seq_len, 512);
        assert!(config.is_umt5());
    }

    #[test]
    fn test_quantized_config() {
        let config = QuantizedT5Config::umt5_xxl();
        assert_eq!(config.vocab_size, 256384);
        assert_eq!(config.d_model, 4096);
    }

    #[test]
    fn test_umt5_vs_t5_config_differences() {
        let t5 = T5EncoderConfig::t5_xxl();
        let umt5 = T5EncoderConfig::umt5_xxl();

        // Same architecture
        assert_eq!(t5.d_model, umt5.d_model);
        assert_eq!(t5.d_ff, umt5.d_ff);
        assert_eq!(t5.num_heads, umt5.num_heads);
        assert_eq!(t5.num_layers, umt5.num_layers);

        // Different vocab size
        assert_ne!(t5.vocab_size, umt5.vocab_size);
        assert_eq!(t5.vocab_size, 32128);
        assert_eq!(umt5.vocab_size, 256384);

        // Different max_seq_len
        assert_eq!(t5.max_seq_len, 256);
        assert_eq!(umt5.max_seq_len, 512);
    }

    #[test]
    fn test_candle_config_conversion() {
        let config = T5EncoderConfig::umt5_xxl();
        let candle_config = config.to_candle_t5_config();

        assert_eq!(candle_config.vocab_size, 256384);
        assert_eq!(candle_config.d_model, 4096);
        assert_eq!(candle_config.d_ff, 10240);
        assert_eq!(candle_config.num_heads, 64);
        assert_eq!(candle_config.num_layers, 24);
    }

    #[test]
    fn test_expected_embedding_shape() {
        // Wan uses [batch, seq_len, 4096] embeddings
        let config = T5EncoderConfig::umt5_xxl();

        // Expected output shape for batch=1, seq_len=512
        let batch_size = 1;
        let seq_len = config.max_seq_len;
        let hidden_dim = config.d_model;

        assert_eq!(batch_size, 1);
        assert_eq!(seq_len, 512);
        assert_eq!(hidden_dim, 4096);
    }
}
