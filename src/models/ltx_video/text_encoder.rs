use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::t5;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::t2v_pipeline::{TextEncoder as VTextEncoder, Tokenizer as VTokenizer};
use crate::interfaces::conditioning::{Conditioning, TextConditioner};
use crate::loader::LoaderError;

pub use crate::interfaces::t5_encoder::T5EncoderConfig as CommonT5EncoderConfig;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5EncoderConfig {
    pub d_model: usize,

    #[serde(default = "default_d_ff")]
    pub d_ff: usize,

    #[serde(default = "default_d_kv")]
    pub d_kv: usize,

    pub num_heads: usize,

    pub num_layers: usize,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,

    #[serde(default = "default_relative_attention_num_buckets")]
    pub relative_attention_num_buckets: usize,

    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,

    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[serde(default)]
    pub cpu_offload: bool,

    #[serde(default)]
    pub dropout_rate: f64,
}

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

    pub fn from_json(path: impl AsRef<Path>) -> std::result::Result<Self, LoaderError> {
        crate::loader::load_model_config(path)
    }

    pub fn with_cpu_offload(mut self, enable: bool) -> Self {
        self.cpu_offload = enable;
        self
    }

    pub fn with_max_seq_len(mut self, max_len: usize) -> Self {
        self.max_seq_len = max_len;
        self
    }

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

pub struct T5TextEncoderWrapper {
    config: T5EncoderConfig,
    device: Device,
    dtype: DType,
    model: Option<t5::T5EncoderModel>,
    cache: EmbeddingCache,
}

impl T5TextEncoderWrapper {
    pub fn new(config: T5EncoderConfig, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            config,
            device,
            dtype,
            model: None,
            cache: EmbeddingCache::new(),
        })
    }

    pub fn load_model(&mut self, vb: candle_nn::VarBuilder) -> Result<()> {
        let candle_config = self.config.to_candle_t5_config();
        let model = t5::T5EncoderModel::load(vb, &candle_config)?;
        self.model = Some(model);
        Ok(())
    }

    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn config(&self) -> &T5EncoderConfig {
        &self.config
    }

    pub fn is_cpu_offload_enabled(&self) -> bool {
        self.config.cpu_offload
    }

    pub fn enable_cache(&mut self, enable: bool) {
        self.cache.enabled = enable;
    }

    pub fn cache_contains(&self, prompt: &str) -> bool {
        self.cache.contains(prompt)
    }

    pub fn cache_hits(&self) -> usize {
        self.cache.hits
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn mock_tokenize(&self, prompt: &str) -> Vec<u32> {
        if prompt.is_empty() {
            return vec![1];
        }

        let words: Vec<&str> = prompt.split_whitespace().collect();
        let mut tokens: Vec<u32> = words
            .iter()
            .enumerate()
            .map(|(i, _)| (i + 2) as u32)
            .collect();

        if tokens.len() >= self.config.max_seq_len {
            tokens.truncate(self.config.max_seq_len - 1);
        }

        tokens.push(1);

        tokens
    }

    pub fn token_ids_to_tensor(&self, token_ids: &[u32]) -> Result<Tensor> {
        let tokens: Vec<u32> = token_ids.to_vec();
        let tensor = Tensor::new(&tokens[..], &self.device)?;
        tensor.unsqueeze(0)
    }

    pub fn encode(&mut self, prompt: &str) -> std::result::Result<Tensor, TextEncoderError> {
        if self.model.is_none() {
            return Err(TextEncoderError::ModelNotLoaded);
        }

        if let Some(cached) = self.cache.get(prompt) {
            return Ok(cached.clone());
        }

        let token_ids = self.mock_tokenize(prompt);
        let input_tensor = self.token_ids_to_tensor(&token_ids)?;

        let model = self.model.as_mut().unwrap();
        let embeddings = model.forward(&input_tensor)?;

        self.cache.insert(prompt.to_string(), embeddings.clone());

        Ok(embeddings)
    }

    pub fn mock_encode(&mut self, prompt: &str) -> Result<Tensor> {
        if let Some(cached) = self.cache.get(prompt) {
            return Ok(cached.clone());
        }

        let token_ids = self.mock_tokenize(prompt);
        let seq_len = token_ids.len();

        let embeddings = Tensor::randn(0f32, 1.0, (1, seq_len, self.config.d_model), &self.device)?;

        self.cache.insert(prompt.to_string(), embeddings.clone());

        Ok(embeddings)
    }

    pub fn mock_encode_batch(&mut self, prompts: &[&str]) -> Result<Tensor> {
        let batch_size = prompts.len();

        let token_batches: Vec<Vec<u32>> = prompts.iter().map(|p| self.mock_tokenize(p)).collect();

        let max_seq_len = token_batches.iter().map(|t| t.len()).max().unwrap_or(1);

        let embeddings = Tensor::randn(
            0f32,
            1.0,
            (batch_size, max_seq_len, self.config.d_model),
            &self.device,
        )?;

        Ok(embeddings)
    }

    pub fn mock_encode_for_cfg(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        let pos_tokens = self.mock_tokenize(prompt);
        let neg_tokens = self.mock_tokenize(negative_prompt);

        let max_len = pos_tokens.len().max(neg_tokens.len());

        let positive_emb =
            Tensor::randn(0f32, 1.0, (1, max_len, self.config.d_model), &self.device)?;

        let negative_emb =
            Tensor::randn(0f32, 0.1, (1, max_len, self.config.d_model), &self.device)?;

        Ok((positive_emb, negative_emb))
    }

    pub fn encode_for_cfg(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> std::result::Result<(Tensor, Tensor), TextEncoderError> {
        let positive_emb = self.encode(prompt)?;
        let negative_emb = self.encode(negative_prompt)?;

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

    fn pad_to_length(&self, embeddings: &Tensor, target_len: usize) -> Result<Tensor> {
        let current_len = embeddings.dim(1)?;

        if current_len >= target_len {
            return Ok(embeddings.clone());
        }

        let batch_size = embeddings.dim(0)?;
        let d_model = embeddings.dim(2)?;
        let pad_len = target_len - current_len;

        let padding = Tensor::zeros((batch_size, pad_len, d_model), self.dtype, &self.device)?;

        Tensor::cat(&[embeddings, &padding], 1)
    }

    pub fn clear_kv_cache(&mut self) {
        if let Some(model) = &mut self.model {
            model.clear_kv_cache();
        }
    }
}

impl VTextEncoder for T5TextEncoderWrapper {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match &mut self.model {
            Some(m) => m.forward(input_ids),
            None => candle_core::bail!("T5 Model not loaded"),
        }
    }
}

impl VTokenizer for T5TextEncoderWrapper {
    fn model_max_length(&self) -> usize {
        self.config.max_seq_len
    }
    fn encode_batch(&self, prompts: &[String], max_length: usize) -> Result<(Tensor, Tensor)> {
        let mut all_ids = vec![];
        let mut max_len = 0;
        for p in prompts {
            let ids = self.mock_tokenize(p);
            max_len = max_len.max(ids.len());
            all_ids.push(ids);
        }
        let max_len = max_len.min(max_length);

        let batch_size = prompts.len();
        let mut ids_vec = vec![0u32; batch_size * max_len];
        let mut mask_vec = vec![0u32; batch_size * max_len];

        for (b, ids) in all_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate().take(max_len) {
                ids_vec[b * max_len + i] = id;
                mask_vec[b * max_len + i] = 1;
            }
        }

        let ids_t = Tensor::new(ids_vec, &self.device)?.reshape((batch_size, max_len))?;
        let mask_t = Tensor::new(mask_vec, &self.device)?.reshape((batch_size, max_len))?;
        Ok((ids_t, mask_t))
    }
}

impl TextConditioner for T5TextEncoderWrapper {
    fn encode_prompt(
        &mut self,
        prompt: &str,
        negative: Option<&str>,
        device: &Device,
    ) -> Result<Conditioning> {
        let max_length = self.model_max_length();
        let (input_ids, attention_mask) = self.encode_batch(&[prompt.to_string()], max_length)?;
        let input_ids = input_ids.to_device(device)?;
        let attention_mask = attention_mask.to_device(device)?;
        let prompt_embeds = self.forward(&input_ids)?.to_device(device)?;

        let mut negative_prompt_embeds = None;
        let mut negative_prompt_attention_mask = None;
        if let Some(neg) = negative {
            let (neg_ids, neg_mask) = self.encode_batch(&[neg.to_string()], max_length)?;
            let neg_ids = neg_ids.to_device(device)?;
            let neg_mask = neg_mask.to_device(device)?;
            let neg_embeds = self.forward(&neg_ids)?.to_device(device)?;
            negative_prompt_embeds = Some(neg_embeds);
            negative_prompt_attention_mask = Some(neg_mask);
        }

        Ok(Conditioning {
            prompt_embeds,
            prompt_attention_mask: attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        })
    }
}

use candle_transformers::models::quantized_t5;
use tokenizers::Tokenizer;

pub struct QuantizedT5Encoder {
    model: quantized_t5::T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
    config: quantized_t5::Config,
    d_model: usize,
    max_seq_len: usize,
}

impl QuantizedT5Encoder {
    pub fn load(
        gguf_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        device: &Device,
        max_seq_len: usize,
    ) -> std::result::Result<Self, TextEncoderError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| TextEncoderError::Tokenizer(e.to_string()))?;

        let config_str = std::fs::read_to_string(config_path.as_ref())
            .map_err(|e| TextEncoderError::Tokenizer(format!("Failed to read config: {}", e)))?;
        let config: quantized_t5::Config = serde_json::from_str(&config_str)
            .map_err(|e| TextEncoderError::Tokenizer(format!("Failed to parse config: {}", e)))?;

        let vb = quantized_t5::VarBuilder::from_gguf(gguf_path.as_ref(), device)?;
        let model = quantized_t5::T5ForConditionalGeneration::load(vb, &config)?;

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

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn tokenize(&self, prompt: &str) -> std::result::Result<Vec<u32>, TextEncoderError> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| TextEncoderError::Tokenizer(e.to_string()))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        if ids.len() > self.max_seq_len {
            ids.truncate(self.max_seq_len);
        }

        Ok(ids)
    }

    pub fn encode(&mut self, prompt: &str) -> std::result::Result<Tensor, TextEncoderError> {
        let token_ids = self.tokenize(prompt)?;

        let input_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;

        let encoder_output = self.model.encode(&input_ids)?;

        Ok(encoder_output)
    }

    pub fn encode_for_cfg(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
    ) -> std::result::Result<(Tensor, Tensor), TextEncoderError> {
        let pos_ids = self.tokenize(prompt)?;
        let neg_ids = self.tokenize(negative_prompt)?;

        let max_len = pos_ids.len().max(neg_ids.len());

        let pos_ids = Self::pad_to_length(pos_ids, max_len, self.config.pad_token_id as u32);
        let neg_ids = Self::pad_to_length(neg_ids, max_len, self.config.pad_token_id as u32);

        let pos_input = Tensor::new(&pos_ids[..], &self.device)?.unsqueeze(0)?;
        let neg_input = Tensor::new(&neg_ids[..], &self.device)?.unsqueeze(0)?;

        let pos_emb = self.model.encode(&pos_input)?;
        let neg_emb = self.model.encode(&neg_input)?;

        Ok((pos_emb, neg_emb))
    }

    fn pad_to_length(mut ids: Vec<u32>, target_len: usize, pad_id: u32) -> Vec<u32> {
        while ids.len() < target_len {
            ids.push(pad_id);
        }
        ids
    }
}

impl VTextEncoder for QuantizedT5Encoder {
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.encode(input_ids)
    }
}

impl VTokenizer for QuantizedT5Encoder {
    fn model_max_length(&self) -> usize {
        self.max_seq_len
    }
    fn encode_batch(&self, prompts: &[String], max_length: usize) -> Result<(Tensor, Tensor)> {
        let mut all_ids = vec![];
        let mut max_len = 0;
        for p in prompts {
            let ids = self.tokenize(p).map_err(candle_core::Error::wrap)?;
            max_len = max_len.max(ids.len());
            all_ids.push(ids);
        }
        let max_len = max_len.min(max_length);

        let batch_size = prompts.len();
        let mut ids_vec = vec![0u32; batch_size * max_len];
        let mut mask_vec = vec![0u32; batch_size * max_len];

        let pad_id = self.config.pad_token_id as u32;

        for (b, ids) in all_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate().take(max_len) {
                ids_vec[b * max_len + i] = id;
                mask_vec[b * max_len + i] = 1;
            }

            for i in ids.len()..max_len {
                ids_vec[b * max_len + i] = pad_id;
            }
        }

        let ids_t = Tensor::new(ids_vec, &self.device)?.reshape((batch_size, max_len))?;
        let mask_t = Tensor::new(mask_vec, &self.device)?.reshape((batch_size, max_len))?;
        Ok((ids_t, mask_t))
    }
}

impl TextConditioner for QuantizedT5Encoder {
    fn encode_prompt(
        &mut self,
        prompt: &str,
        negative: Option<&str>,
        device: &Device,
    ) -> Result<Conditioning> {
        let max_length = self.model_max_length();
        let (input_ids, attention_mask) = self.encode_batch(&[prompt.to_string()], max_length)?;
        let input_ids = input_ids.to_device(device)?;
        let attention_mask = attention_mask.to_device(device)?;
        let prompt_embeds = self.forward(&input_ids)?.to_device(device)?;

        let mut negative_prompt_embeds = None;
        let mut negative_prompt_attention_mask = None;
        if let Some(neg) = negative {
            let (neg_ids, neg_mask) = self.encode_batch(&[neg.to_string()], max_length)?;
            let neg_ids = neg_ids.to_device(device)?;
            let neg_mask = neg_mask.to_device(device)?;
            let neg_embeds = self.forward(&neg_ids)?.to_device(device)?;
            negative_prompt_embeds = Some(neg_embeds);
            negative_prompt_attention_mask = Some(neg_mask);
        }

        Ok(Conditioning {
            prompt_embeds,
            prompt_attention_mask: attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        })
    }
}

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
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[2], 1);
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
