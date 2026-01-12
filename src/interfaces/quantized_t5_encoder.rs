//! Quantized T5/UMT5 Encoder Model for GGUF format.
//!
//! This module provides a T5 encoder-only model that loads from GGUF files.
//! Supports both T5-XXL (LTX-Video) and UMT5-XXL (Wan) through configuration.

use candle_core::{DType, Device, Module, Result, Tensor, quantized::QTensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use std::sync::Arc;

use super::t5_encoder::T5EncoderConfig;

// =============================================================================
// Helpers
// =============================================================================

fn gelu_new(x: &Tensor) -> Result<Tensor> {
    let x_cube = (x.sqr()? * x)?;
    let inner = (x + (x_cube * 0.044715)?)?;
    let inner = (inner * (2.0f64 / std::f64::consts::PI).sqrt())?;
    let tanh = inner.tanh()?;
    (x * 0.5f64)?.broadcast_mul(&(tanh + 1.0)?)
}

// =============================================================================
// Quantized Config (GGUF-specific)
// =============================================================================

/// Configuration for quantized T5 encoder (GGUF-specific).
///
/// For general configuration, use `T5EncoderConfig` from `t5_encoder.rs`.
#[derive(Debug, Clone)]
pub struct QuantizedT5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
}

impl From<&T5EncoderConfig> for QuantizedT5Config {
    fn from(cfg: &T5EncoderConfig) -> Self {
        Self {
            vocab_size: cfg.vocab_size,
            d_model: cfg.d_model,
            d_kv: cfg.d_kv,
            d_ff: cfg.d_ff,
            num_layers: cfg.num_layers,
            num_heads: cfg.num_heads,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            layer_norm_epsilon: cfg.layer_norm_epsilon,
        }
    }
}

impl QuantizedT5Config {
    /// T5-XXL encoder configuration.
    pub fn t5_xxl() -> Self {
        Self::from(&T5EncoderConfig::t5_xxl())
    }

    /// UMT5-XXL encoder configuration.
    pub fn umt5_xxl() -> Self {
        Self::from(&T5EncoderConfig::umt5_xxl())
    }
}

// =============================================================================
// Quantized Linear Layer
// =============================================================================

struct QLinear {
    weight: Arc<QTensor>,
}

impl QLinear {
    fn new(weight: Arc<QTensor>) -> Self {
        Self { weight }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.weight.dequantize(x.device())?;
        let in_dims = x.dims();
        let batch_dims = &in_dims[..in_dims.len() - 1];
        let in_features = in_dims[in_dims.len() - 1];

        let x_flat = x.reshape(((), in_features))?;
        let out = x_flat.matmul(&weight.t()?)?;

        let mut out_shape: Vec<usize> = batch_dims.to_vec();
        out_shape.push(out.dim(1)?);
        out.reshape(out_shape)
    }
}

// =============================================================================
// RMS Layer Normalization
// =============================================================================

/// RMS Layer Normalization for quantized models.
pub struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    fn new_from_quantized(weight: Arc<QTensor>, eps: f64, device: &Device) -> Result<Self> {
        let weight = weight.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dim = x.dim(candle_core::D::Minus1)? as f64;
        let ms = x
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let eps_tensor = Tensor::new(&[self.eps as f32], x.device())?.broadcast_as(ms.shape())?;
        let denom = ms.broadcast_add(&eps_tensor)?.sqrt()?;
        let mut ys = x.broadcast_div(&denom)?;

        let rank = ys.rank();
        let mut shape = vec![1usize; rank];
        shape[rank - 1] = self.weight.dims1()?;
        let w = self.weight.reshape(shape)?;
        ys = ys.broadcast_mul(&w)?;

        Ok(ys)
    }
}

// =============================================================================
// T5 Attention Layer
// =============================================================================

struct T5Attention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    relative_attention_bias: Option<Arc<QTensor>>,
    num_heads: usize,
    d_kv: usize,
}

impl T5Attention {
    fn new(vb: &VarBuilder, block_idx: usize, config: &QuantizedT5Config) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let q = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_q.weight", prefix),
        )?);
        let k = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_k.weight", prefix),
        )?);
        let v = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_v.weight", prefix),
        )?);
        let o = QLinear::new(vb.get(
            (config.d_model, config.num_heads * config.d_kv),
            &format!("{}.attn_o.weight", prefix),
        )?);

        let relative_attention_bias = if block_idx == 0 {
            Some(vb.get(
                (config.relative_attention_num_buckets, config.num_heads),
                &format!("{}.attn_rel_b.weight", prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            q,
            k,
            v,
            o,
            relative_attention_bias,
            num_heads: config.num_heads,
            d_kv: config.d_kv,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let q = self.q.forward(hidden_states)?;
        let k = self.k.forward(hidden_states)?;
        let v = self.v.forward(hidden_states)?;

        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;

        let (mut scores, position_bias_out) = if let Some(bias) = position_bias {
            (scores.broadcast_add(bias)?, Some(bias.clone()))
        } else if let Some(ref rel_bias) = self.relative_attention_bias {
            let bias = self.compute_position_bias(seq_len, hidden_states.device(), rel_bias)?;
            let scores = scores.broadcast_add(&bias)?;
            (scores, Some(bias))
        } else {
            (scores, None)
        };

        if let Some(mask) = attention_mask {
            scores = scores.broadcast_add(mask)?;
        }

        // Softmax on CPU for stability
        let scores_cpu = scores.to_device(&Device::Cpu)?;
        let attn_weights_cpu = candle_nn::ops::softmax_last_dim(&scores_cpu)?;
        let attn_weights = attn_weights_cpu.to_device(hidden_states.device())?;

        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.d_kv,
        ))?;

        let output = self.o.forward(&attn_output)?;
        Ok((output, position_bias_out))
    }

    fn compute_position_bias(
        &self,
        seq_len: usize,
        device: &Device,
        rel_bias: &Arc<QTensor>,
    ) -> Result<Tensor> {
        let cpu = Device::Cpu;

        let context_position = Tensor::arange(0u32, seq_len as u32, &cpu)?;
        let memory_position = Tensor::arange(0u32, seq_len as u32, &cpu)?;

        let context_position = context_position.to_dtype(DType::I64)?.unsqueeze(1)?;
        let memory_position = memory_position.to_dtype(DType::I64)?.unsqueeze(0)?;
        let relative_position = memory_position.broadcast_sub(&context_position)?;

        let buckets = self.relative_position_bucket(&relative_position, 32, 128)?;

        let bias_weights = rel_bias.dequantize(&cpu)?;
        let buckets_flat = buckets.flatten_all()?;
        let values = bias_weights.index_select(&buckets_flat, 0)?;
        let values = values.reshape((seq_len, seq_len, self.num_heads))?;
        let bias = values.permute((2, 0, 1))?;

        let bias = bias.unsqueeze(0)?;
        bias.to_device(device)
    }

    fn relative_position_bucket(
        &self,
        relative_position: &Tensor,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Tensor> {
        let relative_position = relative_position.to_dtype(DType::I64)?;

        let num_buckets_i = num_buckets as i64;
        let half_buckets = num_buckets_i / 2;

        let abs_pos = relative_position.abs()?;

        let max_exact = half_buckets / 2;
        let is_small = abs_pos.lt(max_exact)?;

        let max_distance_f = max_distance as f32;
        let max_exact_f = max_exact as f32;
        let abs_pos_f = abs_pos.to_dtype(DType::F32)?;

        let max_exact_tensor =
            Tensor::new(&[max_exact_f], abs_pos_f.device())?.broadcast_as(abs_pos_f.shape())?;
        let ratio = abs_pos_f.broadcast_div(&max_exact_tensor)?;
        let log_ratio = ratio.log()?;
        let log_max = (max_distance_f / max_exact_f).ln();
        let log_max_tensor =
            Tensor::new(&[log_max], log_ratio.device())?.broadcast_as(log_ratio.shape())?;
        let normalized_log = log_ratio.broadcast_div(&log_max_tensor)?;

        let bucket_range = (half_buckets - max_exact) as f32;
        let range_tensor = Tensor::new(&[bucket_range], normalized_log.device())?
            .broadcast_as(normalized_log.shape())?;
        let log_bucket = normalized_log.broadcast_mul(&range_tensor)?;
        let log_bucket = log_bucket.broadcast_add(&max_exact_tensor)?;

        let max_bucket = (half_buckets - 1) as f32;
        let large_bucket = log_bucket.clamp(0f32, max_bucket)?.to_dtype(DType::I64)?;

        let abs_pos_i = abs_pos.to_dtype(DType::I64)?;
        let bucket = is_small.where_cond(&abs_pos_i, &large_bucket)?;

        let is_positive = relative_position.gt(0i64)?;
        let offset = Tensor::new(&[half_buckets], relative_position.device())?
            .broadcast_as(bucket.shape())?;
        let bucket = is_positive.where_cond(&bucket.broadcast_add(&offset)?, &bucket)?;

        bucket.to_dtype(DType::U32)
    }
}

// =============================================================================
// T5 Feed-Forward Layer
// =============================================================================

struct T5FeedForward {
    up: QLinear,
    gate: QLinear,
    down: QLinear,
}

impl T5FeedForward {
    fn new(vb: &VarBuilder, block_idx: usize, config: &QuantizedT5Config) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let up = QLinear::new(vb.get(
            (config.d_ff, config.d_model),
            &format!("{}.ffn_up.weight", prefix),
        )?);
        let gate = QLinear::new(vb.get(
            (config.d_ff, config.d_model),
            &format!("{}.ffn_gate.weight", prefix),
        )?);
        let down = QLinear::new(vb.get(
            (config.d_model, config.d_ff),
            &format!("{}.ffn_down.weight", prefix),
        )?);

        Ok(Self { up, gate, down })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_out = self.gate.forward(x)?;
        let gate_out = gelu_new(&gate_out)?;
        let up_out = self.up.forward(x)?;
        let hidden = (gate_out * up_out)?;
        self.down.forward(&hidden)
    }
}

// =============================================================================
// T5 Encoder Block
// =============================================================================

struct T5EncoderBlock {
    attention: T5Attention,
    attn_norm: QRmsNorm,
    ffn: T5FeedForward,
    ffn_norm: QRmsNorm,
}

impl T5EncoderBlock {
    fn new(
        vb: &VarBuilder,
        block_idx: usize,
        config: &QuantizedT5Config,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let attention = T5Attention::new(vb, block_idx, config)?;
        let attn_norm = QRmsNorm::new_from_quantized(
            vb.get((config.d_model,), &format!("{}.attn_norm.weight", prefix))?,
            config.layer_norm_epsilon,
            device,
        )?;
        let ffn = T5FeedForward::new(vb, block_idx, config)?;
        let ffn_norm = QRmsNorm::new_from_quantized(
            vb.get((config.d_model,), &format!("{}.ffn_norm.weight", prefix))?,
            config.layer_norm_epsilon,
            device,
        )?;

        Ok(Self {
            attention,
            attn_norm,
            ffn,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed = self.attn_norm.forward(hidden_states)?;
        let (attn_output, position_bias_out) =
            self.attention
                .forward(&normed, position_bias, attention_mask)?;
        let hidden_states = (hidden_states + attn_output)?;

        let normed = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let hidden_states = (hidden_states + ffn_output)?;

        Ok((hidden_states, position_bias_out))
    }
}

// =============================================================================
// Quantized T5 Encoder Model
// =============================================================================

/// Quantized T5 Encoder Model (GGUF format).
///
/// Supports T5-XXL (LTX-Video) and UMT5-XXL (Wan).
pub struct QuantizedT5EncoderModel {
    embedding: Arc<QTensor>,
    blocks: Vec<T5EncoderBlock>,
    final_norm: QRmsNorm,
    device: Device,
    config: QuantizedT5Config,
}

impl QuantizedT5EncoderModel {
    /// Load encoder from GGUF file with T5-XXL config.
    pub fn load(gguf_path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        Self::load_with_config(gguf_path, device, QuantizedT5Config::t5_xxl())
    }

    /// Load encoder from GGUF file with UMT5-XXL config.
    pub fn load_umt5(gguf_path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        Self::load_with_config(gguf_path, device, QuantizedT5Config::umt5_xxl())
    }

    /// Load encoder from GGUF file with custom config.
    pub fn load_with_config(
        gguf_path: impl AsRef<std::path::Path>,
        device: &Device,
        config: QuantizedT5Config,
    ) -> Result<Self> {
        let vb = VarBuilder::from_gguf(gguf_path.as_ref(), device)?;

        let embedding = vb.get((config.vocab_size, config.d_model), "token_embd.weight")?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(T5EncoderBlock::new(&vb, i, &config, device)?);
        }

        let final_norm = QRmsNorm::new_from_quantized(
            vb.get((config.d_model,), "enc.output_norm.weight")?,
            config.layer_norm_epsilon,
            device,
        )?;

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            device: device.clone(),
            config,
        })
    }

    /// Forward pass through encoder.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `attention_mask` - Optional mask [batch, seq_len] (1.0 keep, 0.0 mask)
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let embedding_weights = self.embedding.dequantize(&self.device)?;
        let hidden_states =
            candle_nn::Embedding::new(embedding_weights, self.config.d_model).forward(input_ids)?;

        let mut hidden_states = hidden_states;
        let mut position_bias: Option<Tensor> = None;

        let extended_mask = if let Some(mask) = attention_mask {
            let (b, s) = mask.dims2()?;
            let mask = mask.reshape((b, 1, 1, s))?;
            let mask = mask.to_dtype(DType::F32)?;
            let on = Tensor::ones_like(&mask)?;
            let inv_mask = (on - mask)?;
            let bias = (inv_mask * -1e9f64)?;
            Some(bias)
        } else {
            None
        };

        for block in self.blocks.iter() {
            let (new_hidden, new_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
                extended_mask.as_ref(),
            )?;
            hidden_states = new_hidden;
            position_bias = new_bias;
        }

        self.final_norm.forward(&hidden_states)
    }

    /// Get model dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Check if this is a UMT5 model (based on vocab_size).
    pub fn is_umt5(&self) -> bool {
        self.config.vocab_size > 100000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_config() {
        let config = QuantizedT5Config::t5_xxl();
        assert_eq!(config.vocab_size, 32128);
        assert_eq!(config.d_model, 4096);
    }

    #[test]
    fn test_umt5_config() {
        let config = QuantizedT5Config::umt5_xxl();
        assert_eq!(config.vocab_size, 256384);
        assert_eq!(config.d_model, 4096);
    }
}
