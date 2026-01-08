//! Quantized T5 Encoder Model for GGUF format
//!
//! This module provides a T5 encoder-only model that loads from GGUF files.
//! It's designed for text encoding in LTX-Video and similar models.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, quantized::QTensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use std::sync::Arc;

fn gelu_new(x: &Tensor) -> Result<Tensor> {
    let x_cube = (x.sqr()? * x)?;
    let inner = (x + (x_cube * 0.044715)?)?;
    let inner = (inner * (2.0f64 / std::f64::consts::PI).sqrt())?;
    let tanh = inner.tanh()?;
    (x * 0.5f64)?.broadcast_mul(&(tanh + 1.0)?)
}

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

        // Matrix multiply: x @ weight.T
        // VarBuilder for GGUF loads weights in [out_features, in_features] format.
        // So we transpose to [in_features, out_features] for matmul.
        let out = x_flat.matmul(&weight.t()?)?;

        // Reshape back
        let mut out_shape: Vec<usize> = batch_dims.to_vec();
        out_shape.push(out.dim(1)?);
        out.reshape(out_shape)
    }
}

/// RMS Layer Normalization
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
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
        // add eps as scalar via broadcasting
        let eps_tensor = Tensor::new(&[self.eps as f32], x.device())?.broadcast_as(ms.shape())?;
        let denom = ms.broadcast_add(&eps_tensor)?.sqrt()?;
        let mut ys = x.broadcast_div(&denom)?;

        // Broadcast weight over leading dims
        let rank = ys.rank();
        let mut shape = vec![1usize; rank];
        shape[rank - 1] = self.weight.dims1()?;
        let w = self.weight.reshape(shape)?;
        ys = ys.broadcast_mul(&w)?;

        Ok(ys)
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
        attention_mask: Option<&Tensor>,
        block_idx: usize,
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
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;

        // Add position bias
        let (mut scores, position_bias_out) = if let Some(bias) = position_bias {
            (scores.broadcast_add(bias)?, Some(bias.clone()))
        } else if let Some(ref rel_bias) = self.relative_attention_bias {
            let bias = self.compute_position_bias(seq_len, hidden_states.device(), rel_bias)?;
            
            // Debug: print position bias
            if std::env::var("DEBUG_T5").is_ok() {
                if let Ok(vals) = bias.flatten_all()?.to_vec1::<f32>() {
                    println!("[DEBUG T5] Position bias head 0 top 5x5:");
                    for row in 0..5 {
                        let start = row * seq_len;
                        println!("  {:?}", &vals[start..start+5]);
                    }
                }
            }
            
            let scores = scores.broadcast_add(&bias)?;
            (scores, Some(bias))
        } else {
            (scores, None)
        };

        if let Some(mask) = attention_mask {
            scores = scores.broadcast_add(mask)?;
        }

        // Debug: print scores before softmax
        if std::env::var("DEBUG_T5").is_ok() {
            if let Ok(vals) = scores.flatten_all()?.to_vec1::<f32>() {
                println!("[DEBUG T5] Scores before softmax head 0 top 5x5:");
                for row in 0..5 {
                    let start = row * seq_len;
                    println!("  {:?}", &vals[start..start+5]);
                }
            }
        }

        // Softmax and apply to values
        // Fallback to CPU for softmax if CUDA implementation is missing or unstable
        let scores_cpu = scores.to_device(&Device::Cpu)?;
        let attn_weights_cpu = candle_nn::ops::softmax_last_dim(&scores_cpu)?;
        let attn_weights = attn_weights_cpu.to_device(hidden_states.device())?;
        
        let attn_output = attn_weights.matmul(&v)?;
        
        // Debug Attn matmul V
        if std::env::var("DEBUG_T5").is_ok() && block_idx == 0 {
             let transposed = attn_output.transpose(1, 2)?.reshape((
                batch_size,
                seq_len,
                self.num_heads * self.d_kv,
            ))?;
             if let Ok(v) = transposed.flatten_all()?.to_vec1::<f32>() {
                 println!("[DEBUG T5] Block 0 Attn matmul V first 10: {:?}", &v[..10]);
                 println!("[DEBUG T5] Block 0 Token 36 matmul V first 5: {:?}", &v[36*4096..36*4096+5]);
             }
        }


        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.d_kv,
        ))?;

        // Output projection
        let output = self.o.forward(&attn_output)?;
        
        // Debug O output
        if std::env::var("DEBUG_T5").is_ok() && block_idx == 0 {
             // Print raw O weight values for comparison with Python
             if let Ok(o_weight) = self.o.weight.dequantize(&Device::Cpu) {
                 if let Ok(w) = o_weight.flatten_all()?.to_vec1::<f32>() {
                     println!("[DEBUG T5] Block 0 O weight[0, :5]: {:?}", &w[..5]);
                     println!("[DEBUG T5] Block 0 O weight[:5, 0]: {:?}", 
                         vec![w[0], w[4096], w[4096*2], w[4096*3], w[4096*4]]);
                     println!("[DEBUG T5] Block 0 O weight[100, :5]: {:?}", &w[100*4096..100*4096+5]);
                 }
             }
             if let Ok(v) = output.flatten_all()?.to_vec1::<f32>() {
                 let t36_start = 36 * 4096;
                 println!("[DEBUG T5] Block 0 Token 36 O output (before residual) first 5: {:?}", &v[t36_start..t36_start+5]);
             }
        }


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

        // Compute relative position: memory - context (matches transformers)
        let context_position = context_position.to_dtype(DType::I64)?.unsqueeze(1)?;
        let memory_position = memory_position.to_dtype(DType::I64)?.unsqueeze(0)?;
        let relative_position = memory_position.broadcast_sub(&context_position)?;

        // Compute buckets (on CPU)
        let buckets = self.relative_position_bucket(&relative_position, 32, 128)?;

        // Lookup bias values - weights are [num_buckets, num_heads]
        let bias_weights = rel_bias.dequantize(&cpu)?;
        
        // Debug: print specific bucket weights
        if std::env::var("DEBUG_T5").is_ok() {
            println!("[DEBUG T5] Rust relative bias weights (Bucket x Head):");
            for b_idx in [0, 1, 16, 17] {
                if let Ok(v) = bias_weights.i((b_idx, 0))?.to_vec0::<f32>() {
                    println!("  Bucket {:2} Head 0: {:.6}", b_idx, v);
                }
            }
        }

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

        // Handle positive positions (forward-looking)
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

        // Add offset for positive positions (forward-looking memory relative to context)
        let is_positive = relative_position.gt(0i64)?;
        let offset = Tensor::new(&[half_buckets], relative_position.device())?
            .broadcast_as(bucket.shape())?;
        let bucket = is_positive.where_cond(&bucket.broadcast_add(&offset)?, &bucket)?;

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

        // Candle VarBuilder sees GGUF shapes as [out, in] for linear layers
        // ffn_up/gate: [10240, 4096], ffn_down: [4096, 10240]
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
        // Gated GeLU: down(gelu_new(gate(x)) * up(x))
        // T5-v1.1-xxl uses gelu_new (tanh approximation)
        let gate_out = self.gate.forward(x)?;
        
        // Debug FFN values
        if std::env::var("DEBUG_T5").is_ok() {
             // We can't easily print the full weight here without dequantizing every time,
             // but we can look at the output of the first linear.
             if let Ok(v) = gate_out.flatten_all()?.to_vec1::<f32>() {
                 println!("[DEBUG T5] FFN Gate out first 10: {:?}", &v[..10]);
             }
        }

        let gate_out = gelu_new(&gate_out)?;
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
        let attn_norm = RmsNorm::new_from_quantized(
            vb.get((config.d_model,), &format!("{}.attn_norm.weight", prefix))?,
            config.layer_norm_epsilon,
            device,
        )?;
        let ffn = T5FeedForward::new(vb, block_idx, config)?;
        let ffn_norm = RmsNorm::new_from_quantized(
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
        block_idx: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Self-attention with pre-norm
        let normed = self.attn_norm.forward(hidden_states)?;
        
        // Debug: print after block 0 layer_norm
        if block_idx == 0 && std::env::var("DEBUG_T5").is_ok() {
             if let Ok(vals) = normed.flatten_all()?.to_vec1::<f32>() {
                 println!("[DEBUG T5] After block 0 layer_norm first 10: {:?}", &vals[..10]);
             }
        }

        let (attn_output, position_bias_out) = self.attention.forward(&normed, position_bias, attention_mask, block_idx)?;
        let hidden_states = (hidden_states + attn_output)?;

        // Debug: print after block 0 attention
        if block_idx == 0 && std::env::var("DEBUG_T5").is_ok() {
             if let Ok(vals) = hidden_states.flatten_all()?.to_vec1::<f32>() {
                 println!("[DEBUG T5] After block 0 attention first 10: {:?}", &vals[..10]);
                 println!("[DEBUG T5] After block 0 Token 36 attention first 5: {:?}", &vals[36*4096..36*4096+5]);
             }
        }


        // Feed-forward with pre-norm
        let normed = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let hidden_states = (hidden_states + ffn_output)?;

        // Debug: print after block 0 full
        if block_idx == 0 && std::env::var("DEBUG_T5").is_ok() {
             if let Ok(vals) = hidden_states.flatten_all()?.to_vec1::<f32>() {
                 let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                 println!("[DEBUG T5] After block 0 (full) first 10: {:?}", &vals[..10]);
                 println!("[DEBUG T5] After block 0 mean: {:.6}", mean);
             }
        }

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
        let final_norm = RmsNorm::new_from_quantized(
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
    /// input_ids: [batch, seq_len]
    /// attention_mask: Optional [batch, seq_len] (1.0 for keep, 0.0 for mask) - will be converted to extended mask
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Embed tokens - GGUF stores [vocab_size, d_model]
        let embedding_weights = self.embedding.dequantize(&self.device)?;
        
        let hidden_states =
            candle_nn::Embedding::new(embedding_weights, self.config.d_model).forward(input_ids)?;

        // Debug: print token embeddings
        if std::env::var("DEBUG_T5").is_ok() {
             println!("[DEBUG T5] Input IDs first 20: {:?}", &input_ids.flatten_all()?.to_vec1::<u32>()?[..20]);
             if let Ok(vals) = hidden_states.flatten_all()?.to_vec1::<f32>() {
                 println!("[DEBUG T5] Token embeddings first 10: {:?}", &vals[..10]);
             }
        }

        // Run through encoder blocks
        let mut hidden_states = hidden_states;
        let mut position_bias: Option<Tensor> = None;

        // Prepare extended attention mask if provided
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

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_bias) = block.forward(&hidden_states, position_bias.as_ref(), extended_mask.as_ref(), i)?;
            hidden_states = new_hidden;
            position_bias = new_bias;
        }

        // Final layer norm
        let out = self.final_norm.forward(&hidden_states)?;
        
        // Final debug for Token 36
        if std::env::var("DEBUG_T5").is_ok() {
             if let Ok(vals) = out.flatten_all()?.to_vec1::<f32>() {
                 let t36_start = 36 * 4096;
                 println!("[DEBUG T5] Final Output Token 36 First 10: {:?}", &vals[t36_start..t36_start+10]);
                 if t36_start + 1478 < vals.len() {
                    println!("[DEBUG T5] Final Output Token 36 Pos 1478: {:?}", &vals[t36_start + 1478]);
                 }
             }
        }
        Ok(out)
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }
}
