//! Quantized T5 Encoder Model for GGUF format
//!
//! This module provides a T5 encoder-only model that loads from GGUF files.
//! It's designed for text encoding in LTX-Video and similar models.

use candle_core::{DType, Device, Module, Result, Tensor, quantized::QTensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use std::sync::Arc;

/// Configuration for quantized T5 encoder
#[derive(Debug, Clone)]
pub struct T5EncoderConfig {
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

impl T5EncoderConfig {
    /// T5-XXL encoder configuration
    pub fn t5_xxl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
        }
    }
}

/// Quantized linear layer using QTensor
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

        // Flatten batch dimensions
        let x_flat = x.reshape(((), in_features))?;

        // Matrix multiply
        let out = x_flat.matmul(&weight.t()?)?;

        // Reshape back
        let mut out_shape: Vec<usize> = batch_dims.to_vec();
        out_shape.push(out.dim(1)?);
        out.reshape(out_shape)
    }
}

/// RMS Layer Normalization
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Arc<QTensor>, eps: f64, device: &Device) -> Result<Self> {
        let weight = weight.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps_tensor =
            Tensor::new(&[self.eps as f32], x.device())?.broadcast_as(variance.shape())?;
        let x_norm = x.broadcast_div(&(variance.broadcast_add(&eps_tensor)?.sqrt()?))?;
        x_norm.broadcast_mul(&self.weight)
    }
}

/// T5 Attention layer
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
    fn new(vb: &VarBuilder, block_idx: usize, config: &T5EncoderConfig) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let q = QLinear::new(vb.get(
            (config.d_model, config.num_heads * config.d_kv),
            &format!("{}.attn_q.weight", prefix),
        )?);
        let k = QLinear::new(vb.get(
            (config.d_model, config.num_heads * config.d_kv),
            &format!("{}.attn_k.weight", prefix),
        )?);
        let v = QLinear::new(vb.get(
            (config.d_model, config.num_heads * config.d_kv),
            &format!("{}.attn_v.weight", prefix),
        )?);
        let o = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_o.weight", prefix),
        )?);

        // Relative position bias only in first layer - shape (32, 64) = (num_buckets, num_heads)
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
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Compute Q, K, V
        let q = self.q.forward(hidden_states)?;
        let k = self.k.forward(hidden_states)?;
        let v = self.v.forward(hidden_states)?;

        // Reshape to [batch, heads, seq, d_kv]
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

        // Attention scores
        let scale = (self.d_kv as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores / scale)?;

        // Add position bias
        let (scores, position_bias_out) = if let Some(bias) = position_bias {
            (scores.broadcast_add(bias)?, Some(bias.clone()))
        } else if let Some(ref rel_bias) = self.relative_attention_bias {
            let bias = self.compute_position_bias(seq_len, hidden_states.device(), rel_bias)?;
            let scores = scores.broadcast_add(&bias)?;
            (scores, Some(bias))
        } else {
            (scores, None)
        };

        // Softmax and apply to values
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.d_kv,
        ))?;

        // Output projection
        let output = self.o.forward(&attn_output)?;

        Ok((output, position_bias_out))
    }

    fn compute_position_bias(
        &self,
        seq_len: usize,
        device: &Device,
        rel_bias: &Arc<QTensor>,
    ) -> Result<Tensor> {
        // Compute position indices on CPU to avoid CUDA kernel issues with I64 operations
        let cpu = Device::Cpu;

        // Create position indices on CPU
        let context_position = Tensor::arange(0u32, seq_len as u32, &cpu)?;
        let memory_position = Tensor::arange(0u32, seq_len as u32, &cpu)?;

        // Compute relative position: context - memory (on CPU as i32 to avoid I64 issues)
        let context_position = context_position.to_dtype(DType::I64)?.unsqueeze(1)?;
        let memory_position = memory_position.to_dtype(DType::I64)?.unsqueeze(0)?;
        let relative_position = context_position.broadcast_sub(&memory_position)?;

        // Compute buckets (on CPU)
        let buckets = self.relative_position_bucket(&relative_position, 32, 128)?;

        // Lookup bias values - weights are [num_buckets, num_heads]
        let bias_weights = rel_bias.dequantize(&cpu)?;
        let buckets_flat = buckets.flatten_all()?;
        // index_select on dim 0: [selected, num_heads]
        let values = bias_weights.index_select(&buckets_flat, 0)?;
        // Reshape to [seq_len, seq_len, num_heads]
        let values = values.reshape((seq_len, seq_len, self.num_heads))?;
        // Permute to [num_heads, seq_len, seq_len]
        let bias = values.permute((2, 0, 1))?;

        // Add batch dimension and move to target device
        let bias = bias.unsqueeze(0)?;
        bias.to_device(device)
    }

    fn relative_position_bucket(
        &self,
        relative_position: &Tensor,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Tensor> {
        // Simplified bucket computation for encoder (bidirectional)
        let relative_position = relative_position.to_dtype(DType::I64)?;

        let num_buckets_i = num_buckets as i64;
        let half_buckets = num_buckets_i / 2;

        // Handle negative positions
        let is_negative = relative_position.lt(0i64)?;
        let abs_pos = relative_position.abs()?;

        // Compute bucket for small positions
        let max_exact = half_buckets / 2;
        let is_small = abs_pos.lt(max_exact)?;

        // For larger positions, use log scale
        let max_distance_f = max_distance as f32;
        let max_exact_f = max_exact as f32;
        let abs_pos_f = abs_pos.to_dtype(DType::F32)?;

        // log_ratio = log(abs_pos / max_exact) / log(max_distance / max_exact)
        let max_exact_tensor =
            Tensor::new(&[max_exact_f], abs_pos_f.device())?.broadcast_as(abs_pos_f.shape())?;
        let ratio = abs_pos_f.broadcast_div(&max_exact_tensor)?;
        let log_ratio = ratio.log()?;
        let log_max = (max_distance_f / max_exact_f).ln();
        let log_max_tensor =
            Tensor::new(&[log_max], log_ratio.device())?.broadcast_as(log_ratio.shape())?;
        let normalized_log = log_ratio.broadcast_div(&log_max_tensor)?;

        // log_bucket = normalized_log * (half_buckets - max_exact) + max_exact
        let bucket_range = (half_buckets - max_exact) as f32;
        let range_tensor = Tensor::new(&[bucket_range], normalized_log.device())?
            .broadcast_as(normalized_log.shape())?;
        let log_bucket = normalized_log.broadcast_mul(&range_tensor)?;
        let log_bucket = log_bucket.broadcast_add(&max_exact_tensor)?;

        // Clamp to valid range
        let max_bucket = (half_buckets - 1) as f32;
        let large_bucket = log_bucket.clamp(0f32, max_bucket)?.to_dtype(DType::I64)?;

        // Combine small and large buckets
        let abs_pos_i = abs_pos.to_dtype(DType::I64)?;
        let bucket = is_small.where_cond(&abs_pos_i, &large_bucket)?;

        // Add offset for negative positions
        let offset = Tensor::new(&[half_buckets], relative_position.device())?
            .broadcast_as(bucket.shape())?;
        let bucket = is_negative.where_cond(&bucket.broadcast_add(&offset)?, &bucket)?;

        bucket.to_dtype(DType::U32)
    }
}

/// T5 Feed-Forward layer (gated)
struct T5FeedForward {
    up: QLinear,
    gate: QLinear,
    down: QLinear,
}

impl T5FeedForward {
    fn new(vb: &VarBuilder, block_idx: usize, config: &T5EncoderConfig) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        // Verified shapes: ffn_up/gate=(10240, 4096), ffn_down=(4096, 10240)
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
        // Gated GeLU: down(gelu(gate(x)) * up(x))
        let gate_out = self.gate.forward(x)?;
        let gate_out = gate_out.gelu_erf()?;
        let up_out = self.up.forward(x)?;
        let hidden = (gate_out * up_out)?;
        self.down.forward(&hidden)
    }
}

/// T5 Encoder Block
struct T5EncoderBlock {
    attention: T5Attention,
    attn_norm: RmsNorm,
    ffn: T5FeedForward,
    ffn_norm: RmsNorm,
}

impl T5EncoderBlock {
    fn new(
        vb: &VarBuilder,
        block_idx: usize,
        config: &T5EncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let attention = T5Attention::new(vb, block_idx, config)?;
        let attn_norm = RmsNorm::new(
            vb.get((config.d_model,), &format!("{}.attn_norm.weight", prefix))?,
            config.layer_norm_epsilon,
            device,
        )?;
        let ffn = T5FeedForward::new(vb, block_idx, config)?;
        let ffn_norm = RmsNorm::new(
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
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Self-attention with pre-norm
        let normed = self.attn_norm.forward(hidden_states)?;
        let (attn_output, position_bias_out) = self.attention.forward(&normed, position_bias)?;
        let hidden_states = (hidden_states + attn_output)?;

        // Feed-forward with pre-norm
        let normed = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let hidden_states = (hidden_states + ffn_output)?;

        Ok((hidden_states, position_bias_out))
    }
}

/// Quantized T5 Encoder Model
pub struct QuantizedT5EncoderModel {
    embedding: Arc<QTensor>,
    blocks: Vec<T5EncoderBlock>,
    final_norm: RmsNorm,
    device: Device,
    config: T5EncoderConfig,
}

impl QuantizedT5EncoderModel {
    /// Load encoder from GGUF file
    pub fn load(gguf_path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let config = T5EncoderConfig::t5_xxl();
        Self::load_with_config(gguf_path, device, config)
    }

    /// Load encoder from GGUF file with custom config
    pub fn load_with_config(
        gguf_path: impl AsRef<std::path::Path>,
        device: &Device,
        config: T5EncoderConfig,
    ) -> Result<Self> {
        let vb = VarBuilder::from_gguf(gguf_path.as_ref(), device)?;

        // Load embedding - GGUF stores as [vocab_size, d_model]
        let embedding = vb.get((config.vocab_size, config.d_model), "token_embd.weight")?;

        // Load encoder blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(T5EncoderBlock::new(&vb, i, &config, device)?);
        }

        // Load final layer norm
        let final_norm = RmsNorm::new(
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

    /// Forward pass through encoder
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Embed tokens - GGUF stores [vocab_size, d_model]
        let embedding_weights = self.embedding.dequantize(&self.device)?;
        let hidden_states =
            candle_nn::Embedding::new(embedding_weights, self.config.d_model).forward(input_ids)?;

        // Run through encoder blocks
        let mut hidden_states = hidden_states;
        let mut position_bias: Option<Tensor> = None;

        for block in &self.blocks {
            let (new_hidden, new_bias) = block.forward(&hidden_states, position_bias.as_ref())?;
            hidden_states = new_hidden;
            position_bias = new_bias;
        }

        // Final layer norm
        self.final_norm.forward(&hidden_states)
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }
}
