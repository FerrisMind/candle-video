//! Memory-efficient operations for video generation
//!
//! This module provides memory-optimized implementations for attention and other
//! computationally expensive operations. Key optimizations:
//!
//! - Chunked attention: Process attention in chunks to reduce peak memory
//! - Separate CFG passes: Process positive and negative separately instead of batched
//! - Aggressive tensor dropping: Explicitly drop intermediate tensors

use candle_core::{D, Result, Tensor};
use candle_nn::ops::softmax;

/// Configuration for memory-efficient attention
#[derive(Debug, Clone)]
pub struct MemoryEfficientConfig {
    /// Number of query chunks to process at once
    /// Smaller = less memory, more iterations
    pub query_chunk_size: usize,

    /// Number of key-value chunks to process at once
    pub kv_chunk_size: usize,

    /// Whether to use chunked attention (disable for small sequences)
    pub enable_chunked_attention: bool,

    /// Threshold sequence length above which chunked attention is used
    pub chunk_threshold: usize,
}

impl Default for MemoryEfficientConfig {
    fn default() -> Self {
        Self {
            query_chunk_size: 1024,
            kv_chunk_size: 1024,
            enable_chunked_attention: true,
            chunk_threshold: 4096,
        }
    }
}

impl MemoryEfficientConfig {
    /// Configuration for low memory usage (slower)
    pub fn low_memory() -> Self {
        Self {
            query_chunk_size: 512,
            kv_chunk_size: 512,
            enable_chunked_attention: true,
            chunk_threshold: 2048,
        }
    }

    /// Configuration for balanced memory/speed
    pub fn balanced() -> Self {
        Self {
            query_chunk_size: 1024,
            kv_chunk_size: 1024,
            enable_chunked_attention: true,
            chunk_threshold: 4096,
        }
    }

    /// Configuration for maximum speed (may use more memory)
    pub fn high_performance() -> Self {
        Self {
            query_chunk_size: 2048,
            kv_chunk_size: 2048,
            enable_chunked_attention: true,
            chunk_threshold: 8192,
        }
    }
}

/// Memory-efficient scaled dot-product attention using chunking
///
/// Instead of computing the full attention matrix at once, this function
/// processes the query sequence in chunks, significantly reducing peak memory usage.
///
/// For a sequence of length N with chunk size C:
/// - Standard attention: O(N²) memory for attention matrix
/// - Chunked attention: O(N × C) memory per chunk
///
/// # Arguments
/// * `query` - Query tensor [batch, heads, seq_q, head_dim]
/// * `key` - Key tensor [batch, heads, seq_kv, head_dim]  
/// * `value` - Value tensor [batch, heads, seq_kv, head_dim]
/// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
/// * `attention_mask` - Optional attention mask
/// * `config` - Memory efficiency configuration
pub fn chunked_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f64,
    attention_mask: Option<&Tensor>,
    config: &MemoryEfficientConfig,
) -> Result<Tensor> {
    let (_batch_size, _num_heads, seq_q, _head_dim) = query.dims4()?;
    let _seq_kv = key.dim(2)?;

    // Use standard attention for small sequences
    if !config.enable_chunked_attention || seq_q <= config.chunk_threshold {
        return standard_attention(query, key, value, scale, attention_mask);
    }

    let query_chunk_size = config.query_chunk_size.min(seq_q);
    let num_query_chunks = seq_q.div_ceil(query_chunk_size);

    // Pre-transpose key for efficiency: [batch, heads, head_dim, seq_kv]
    let key_t = key.transpose(2, 3)?.contiguous()?;

    let mut output_chunks = Vec::with_capacity(num_query_chunks);

    for chunk_idx in 0..num_query_chunks {
        let start = chunk_idx * query_chunk_size;
        let end = ((chunk_idx + 1) * query_chunk_size).min(seq_q);
        let chunk_len = end - start;

        // Extract query chunk: [batch, heads, chunk_len, head_dim]
        let q_chunk = query.narrow(2, start, chunk_len)?;

        // Compute attention scores for this chunk: [batch, heads, chunk_len, seq_kv]
        let attn_weights = q_chunk.matmul(&key_t)?.affine(scale, 0.0)?;

        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            // Extract corresponding mask rows if needed
            let mask_chunk = if mask.dim(2)? > 1 {
                mask.narrow(2, start, chunk_len)?
            } else {
                mask.clone()
            };
            attn_weights.broadcast_add(&mask_chunk)?
        } else {
            attn_weights
        };

        // Softmax and attend
        let attn_probs = softmax(&attn_weights, D::Minus1)?;
        drop(attn_weights); // Explicitly free memory

        let output_chunk = attn_probs.matmul(value)?;
        drop(attn_probs); // Explicitly free memory

        output_chunks.push(output_chunk);
    }

    // Concatenate all output chunks
    Tensor::cat(&output_chunks, 2)
}

/// Standard scaled dot-product attention
fn standard_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f64,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let key_t = key.transpose(2, 3)?.contiguous()?;
    let attn_weights = query.matmul(&key_t)?.affine(scale, 0.0)?;

    let attn_weights = if let Some(mask) = attention_mask {
        attn_weights.broadcast_add(mask)?
    } else {
        attn_weights
    };

    let attn_probs = softmax(&attn_weights, D::Minus1)?;
    attn_probs.matmul(value)
}

/// Memory-efficient attention with KV chunking (for very long key-value sequences)
///
/// This is useful when the KV sequence is very long (e.g., video frames).
/// Processes KV in chunks and accumulates attention using online softmax algorithm.
///
/// Note: This is a simplified version that works for many cases but may have
/// numerical precision issues for very large sequences. For production use,
/// consider using flash-attention.
pub fn chunked_attention_kv(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f64,
    config: &MemoryEfficientConfig,
) -> Result<Tensor> {
    let (_batch_size, _num_heads, _seq_q, _head_dim) = query.dims4()?;
    let seq_kv = key.dim(2)?;

    // Use standard attention for small sequences
    if !config.enable_chunked_attention || seq_kv <= config.chunk_threshold {
        return standard_attention(query, key, value, scale, None);
    }

    // For very long KV sequences, we use a simpler chunking approach:
    // Process each KV chunk separately and concatenate results,
    // then do a final re-normalization pass.
    // This is less memory efficient but simpler and more numerically stable.
    let kv_chunk_size = config.kv_chunk_size.min(seq_kv);
    let num_kv_chunks = seq_kv.div_ceil(kv_chunk_size);

    let mut weighted_sums = Vec::with_capacity(num_kv_chunks);
    let mut sum_weights = Vec::with_capacity(num_kv_chunks);

    for chunk_idx in 0..num_kv_chunks {
        let start = chunk_idx * kv_chunk_size;
        let end = ((chunk_idx + 1) * kv_chunk_size).min(seq_kv);
        let chunk_len = end - start;

        // Extract KV chunk
        let k_chunk = key.narrow(2, start, chunk_len)?;
        let v_chunk = value.narrow(2, start, chunk_len)?;

        // Compute attention scores: [batch, heads, seq_q, chunk_len]
        let k_chunk_t = k_chunk.transpose(2, 3)?;
        let scores = query.matmul(&k_chunk_t)?.affine(scale, 0.0)?;

        // Get max for numerical stability
        let chunk_max = scores.max_keepdim(D::Minus1)?;
        let stable_scores = (&scores - &chunk_max)?;

        // Compute exp and weighted sum
        let exp_scores = stable_scores.exp()?;
        let chunk_sum = exp_scores.sum_keepdim(D::Minus1)?;
        let weighted_v = exp_scores.matmul(&v_chunk)?;

        weighted_sums.push((weighted_v, chunk_max.clone()));
        sum_weights.push((chunk_sum, chunk_max));
    }

    // Combine chunks using log-sum-exp
    // For simplicity, we use a two-pass approach:
    // 1. Find global max
    // 2. Reweight each chunk's contribution

    // Find global max
    let all_maxes: Vec<_> = sum_weights.iter().map(|(_, m)| m.clone()).collect();
    let global_max = if all_maxes.len() == 1 {
        all_maxes[0].clone()
    } else {
        let mut acc = all_maxes[0].clone();
        for m in &all_maxes[1..] {
            acc = acc.maximum(m)?;
        }
        acc
    };

    // Reweight and sum
    let mut total_output = None;
    let mut total_weight = None;

    for ((weighted_v, local_max), (sum_w, _)) in weighted_sums.into_iter().zip(sum_weights) {
        let scale_factor = (&local_max - &global_max)?.exp()?;
        let scaled_v = weighted_v.broadcast_mul(&scale_factor)?;
        let scaled_w = sum_w.broadcast_mul(&scale_factor)?;

        total_output = Some(match total_output {
            Some(acc) => (&acc + &scaled_v)?,
            None => scaled_v,
        });
        total_weight = Some(match total_weight {
            Some(acc) => (&acc + &scaled_w)?,
            None => scaled_w,
        });
    }

    // Normalize
    let output = total_output.unwrap();
    let weight = total_weight.unwrap();
    output.broadcast_div(&weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_chunked_attention_matches_standard() -> Result<()> {
        let device = Device::Cpu;
        let _dtype = DType::F32;

        let batch = 1;
        let heads = 4;
        let seq = 128;
        let head_dim = 64;

        let query = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let key = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let value = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Standard attention
        let standard_output = standard_attention(&query, &key, &value, scale, None)?;

        // Chunked attention with small chunks
        let config = MemoryEfficientConfig {
            query_chunk_size: 32,
            kv_chunk_size: 32,
            enable_chunked_attention: true,
            chunk_threshold: 0, // Force chunking
        };
        let chunked_output = chunked_attention(&query, &key, &value, scale, None, &config)?;

        // Compare outputs (should be nearly identical)
        let diff = (&standard_output - &chunked_output)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "Chunked attention output differs by {}", diff);

        Ok(())
    }
}
