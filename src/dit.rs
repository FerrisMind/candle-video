//! Diffusion Transformer (DiT) implementation for LTX-Video
//!
//! This module implements the core DiT architecture following the LTX-Video paper:
//! - BasicTransformerBlock with AdaLN-single modulation
//! - Self-attention and cross-attention with RoPE
//! - GEGLU-based feed-forward network
//! - Transformer3DModel for full video diffusion

use crate::config::DitConfig;
use crate::rope::FractionalRoPE;
use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias,
};

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ===========================================================================
// Skip Layer Strategy (for STG - Spatio-Temporal Guidance)
// ===========================================================================

/// Strategy for skip layer guidance (STG)
/// Matches Python ltx_video/utils/skip_layer_strategy.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipLayerStrategy {
    /// Skip attention computation, blend output with original hidden states
    AttentionSkip,
    /// Skip attention but use values directly (for STG with value guidance)
    AttentionValues,
    /// Skip at residual level
    Residual,
    /// Skip entire transformer block
    TransformerBlock,
}

// ===========================================================================
// GELU Activation (gelu-approximate)
// ===========================================================================

/// GELU activation layer with linear projection
/// Used in LTX-Video FeedForward with activation_fn="gelu-approximate"
/// Structure: Linear(dim → inner_dim) + GELU(tanh)
pub struct GELUProj {
    proj: Linear,
}

impl GELUProj {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        // Single projection (NOT doubled like GEGLU)
        let proj = linear(in_dim, out_dim, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.proj.forward(x)?;
        // GELU with tanh approximation (gelu-approximate)
        hidden.gelu()
    }
}

// ===========================================================================
// FeedForward Network
// ===========================================================================

/// Feed-forward network with GELU activation
/// Matches diffusers FeedForward(activation_fn="gelu-approximate")
///
/// Structure: net.0 (GELU proj) → net.2 (Linear out)
/// Weight names:
/// - net.0.proj.weight, net.0.proj.bias: Linear(hidden_size, inner_dim)
/// - net.2.weight, net.2.bias: Linear(inner_dim, hidden_size)
pub struct FeedForward {
    gelu_proj: GELUProj,
    proj_out: Linear,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        // LTX-Video uses 4x hidden dimension for inner_dim
        let inner_dim = (config.hidden_size as f64 * config.mlp_ratio) as usize;

        // net.0 = GELU(hidden_size → inner_dim) with gelu-approximate
        let gelu_proj = GELUProj::new(vb.pp("net.0"), config.hidden_size, inner_dim)?;
        // net.2 = Linear(inner_dim → hidden_size)
        let proj_out = linear(inner_dim, config.hidden_size, vb.pp("net.2"))?;

        Ok(Self {
            gelu_proj,
            proj_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // GELU proj → output projection
        let x = self.gelu_proj.forward(x)?;
        self.proj_out.forward(&x)
    }
}

// ===========================================================================
// RMS Norm
// ===========================================================================

/// Root Mean Square Layer Normalization
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        // Load weight from file (should be BF16 natively) - no F32 fallback
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute RMS normalization natively in input dtype (BF16)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        // Use affine_add to add eps while preserving dtype (BF16)
        let var_eps = variance.affine(1.0, self.eps)?;
        let x_normed = x.broadcast_div(&var_eps.sqrt()?)?;
        // Force weight to match x_normed dtype
        let weight = self.weight.to_dtype(x_normed.dtype())?;
        x_normed.broadcast_mul(&weight)
    }
}

// ===========================================================================
// RMS Norm (No Weight) - for elementwise_affine=False
// ===========================================================================

/// Root Mean Square Layer Normalization without learnable weights
/// Used in LTX-Video for norm1/norm2 where elementwise_affine=False
pub struct RMSNormNoWeight {
    eps: f64,
}

impl RMSNormNoWeight {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute RMS normalization natively in input dtype (BF16)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        // Use affine to add eps while preserving dtype (BF16)
        let var_eps = variance.affine(1.0, self.eps)?;
        x.broadcast_div(&var_eps.sqrt()?)
    }
}

// ===========================================================================
// QK Normalization
// ===========================================================================

/// Query-Key normalization (either LayerNorm or RMSNorm)
pub enum QKNorm {
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
    Identity,
}

impl QKNorm {
    pub fn new(vb: VarBuilder, dim: usize, norm_type: Option<&str>) -> Result<Self> {
        match norm_type {
            Some("layer_norm") => {
                let ln = layer_norm(dim, LayerNormConfig::default(), vb)?;
                Ok(Self::LayerNorm(ln))
            }
            Some("rms_norm") => {
                let rms = RMSNorm::new(vb, dim, 1e-5)?;
                Ok(Self::RMSNorm(rms))
            }
            _ => Ok(Self::Identity),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(ln) => ln.forward(x),
            Self::RMSNorm(rms) => rms.forward(x),
            Self::Identity => Ok(x.clone()),
        }
    }
}

// ===========================================================================
// Attention
// ===========================================================================

/// Multi-head attention with optional cross-attention and RoPE support
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: QKNorm,
    k_norm: QKNorm,
    heads: usize,
    head_dim: usize,
    scale: f64,
    #[allow(dead_code)]
    is_cross_attention: bool,
    use_rope: bool,
    use_flash_attn: bool,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        config: &DitConfig,
        cross_attention_dim: Option<usize>,
    ) -> Result<Self> {
        let query_dim = config.hidden_size;
        let inner_dim = config.hidden_size;
        let heads = config.num_heads;
        let head_dim = inner_dim / heads;

        let cross_dim = cross_attention_dim.unwrap_or(query_dim);

        let to_q = linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(cross_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(cross_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out.0"))?;

        // QK normalization (optional, used in LTX-Video)
        let q_norm = QKNorm::new(vb.pp("q_norm"), inner_dim, Some("rms_norm"))?;
        let k_norm = QKNorm::new(vb.pp("k_norm"), inner_dim, Some("rms_norm"))?;

        let scale = (head_dim as f64).powf(-0.5);
        let is_cross_attention = cross_attention_dim.is_some();

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            q_norm,
            k_norm,
            heads,
            head_dim,
            scale,
            is_cross_attention,
            use_rope: !is_cross_attention, // RoPE only for self-attention
            use_flash_attn: config.use_flash_attention,
        })
    }

    /// Apply rotary position embedding (multi-head format: batch, heads, seq, head_dim)
    #[allow(dead_code)]
    fn apply_rotary_emb(&self, x: &Tensor, freqs_cis: &(Tensor, Tensor)) -> Result<Tensor> {
        let (cos_freqs, sin_freqs) = freqs_cis;

        // x: (batch, heads, seq_len, head_dim)
        // freqs: (batch, seq_len, head_dim)

        // Reshape x for rotation: (..., d r) where r=2
        let x_shape = x.dims();
        let batch = x_shape[0];
        let heads = x_shape[1];
        let seq_len = x_shape[2];
        let head_dim = x_shape[3];

        // Pair up dimensions: (batch, heads, seq_len, head_dim/2, 2)
        let x_reshaped = x.reshape((batch, heads, seq_len, head_dim / 2, 2))?;

        // Get x1 and x2
        let x1 = x_reshaped.i((.., .., .., .., 0))?;
        let x2 = x_reshaped.i((.., .., .., .., 1))?;

        // Create rotated version: (-x2, x1)
        let x_rot = Tensor::stack(&[x2.neg()?, x1], D::Minus1)?;
        let x_rot = x_rot.flatten_from(D::Minus2)?; // (batch, heads, seq_len, head_dim)

        // Expand freqs to match attention shape: (batch, 1, seq_len, head_dim)
        let cos_expanded = cos_freqs.unsqueeze(1)?;
        let sin_expanded = sin_freqs.unsqueeze(1)?;

        // Apply rotation: x * cos + x_rot * sin
        let out = x
            .broadcast_mul(&cos_expanded)?
            .add(&x_rot.broadcast_mul(&sin_expanded)?)?;

        Ok(out)
    }

    /// Apply rotary position embedding (linear format: batch, seq, inner_dim)
    /// This matches diffusers apply_rotary_emb which operates on linear Q/K before reshaping to heads
    fn apply_rotary_emb_linear(&self, x: &Tensor, freqs_cis: &(Tensor, Tensor)) -> Result<Tensor> {
        let (cos_freqs, sin_freqs) = freqs_cis;

        // x: (batch, seq_len, inner_dim) where inner_dim = heads * head_dim
        // freqs: (batch, seq_len, inner_dim)
        let x_shape = x.dims();
        let batch = x_shape[0];
        let seq_len = x_shape[1];
        let inner_dim = x_shape[2];

        // Pair up dimensions: (batch, seq_len, inner_dim/2, 2)
        let x_reshaped = x.reshape((batch, seq_len, inner_dim / 2, 2))?;

        // Get real and imag parts (matching diffusers x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1))
        let x_real = x_reshaped.i((.., .., .., 0))?; // (batch, seq_len, inner_dim/2)
        let x_imag = x_reshaped.i((.., .., .., 1))?;

        // Create rotated version: (-x_imag, x_real) then flatten
        // Matches: x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        let x_rot = Tensor::stack(&[x_imag.neg()?, x_real], D::Minus1)?;
        let x_rot = x_rot.flatten_from(D::Minus2)?; // (batch, seq_len, inner_dim)

        // Apply rotation: x * cos + x_rot * sin
        let out = x
            .broadcast_mul(cos_freqs)?
            .add(&x_rot.broadcast_mul(sin_freqs)?)?;

        Ok(out)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        freqs_cis: Option<&(Tensor, Tensor)>,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
        skip_layer_strategy: Option<SkipLayerStrategy>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Compute Q, K, V
        let query = self.to_q.forward(hidden_states)?;
        let mut query = self.q_norm.forward(&query)?;

        let (mut key, value) = if let Some(enc_hidden) = encoder_hidden_states {
            // Cross-attention
            let k = self.to_k.forward(enc_hidden)?;
            let v = self.to_v.forward(enc_hidden)?;
            (self.k_norm.forward(&k)?, v)
        } else {
            // Self-attention
            let k = self.to_k.forward(hidden_states)?;
            let v = self.to_v.forward(hidden_states)?;
            (self.k_norm.forward(&k)?, v)
        };

        // Store value for STG before reshaping (used by AttentionValues strategy)
        let value_for_stg = value.clone();

        // Apply RoPE BEFORE reshape to heads (matching diffusers)
        // RoPE is only applied for self-attention, on linear Q/K format
        if self.use_rope
            && encoder_hidden_states.is_none()
            && let Some(freqs) = freqs_cis
        {
            query = self.apply_rotary_emb_linear(&query, freqs)?;
            key = self.apply_rotary_emb_linear(&key, freqs)?;
        }

        // Reshape to multi-head format: (batch, seq, inner_dim) -> (batch, heads, seq, head_dim)
        // This is done AFTER RoPE, matching diffusers: query.unflatten(2, (attn.heads, -1))
        let query = query
            .reshape((batch_size, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_seq_len = key.dim(1)?;
        let key = key
            .reshape((batch_size, key_seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = value
            .reshape((batch_size, key_seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let use_flash_attn = self.use_flash_attn
            && attention_mask.is_none()
            && matches!(hidden_states.device(), Device::Cuda(_))
            && matches!(query.dtype(), DType::F16 | DType::BF16);

        // Scaled dot-product attention
        // Q: [batch, heads, seq_q, head_dim]
        // K: [batch, heads, seq_kv, head_dim]
        // V: [batch, heads, seq_kv, head_dim]
        let attn_output = if use_flash_attn {
            // flash-attn expects (batch, seq, heads, head_dim)
            let q = query.transpose(1, 2)?;
            let k = key.transpose(1, 2)?;
            let v = value.transpose(1, 2)?;
            flash_attn(&q, &k, &v, self.scale as f32, false)?.transpose(1, 2)?
        } else {
            // K^T: [batch, heads, head_dim, seq_kv]
            let key_t = key.transpose(2, 3)?.contiguous()?;
            // Use affine to multiply by scale while preserving dtype (BF16)
            let attn_weights = query.matmul(&key_t)?.affine(self.scale, 0.0)?;

            // Apply attention mask if provided
            let attn_weights = if let Some(mask) = attention_mask {
                attn_weights.broadcast_add(mask)?
            } else {
                attn_weights
            };

            let attn_probs = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

            // Apply attention to values: [batch, heads, seq_q, seq_kv] @ [batch, heads, seq_kv, head_dim]
            // Result: [batch, heads, seq_q, head_dim]
            attn_probs.matmul(&value)?
        };

        // Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden_dim)
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.heads * self.head_dim,
        ))?;

        // Apply skip layer mask for STG (Spatio-Temporal Guidance)
        // Matches Python attention.py lines 1071-1086
        let hidden_states_out =
            if let (Some(mask), Some(strategy)) = (skip_layer_mask, skip_layer_strategy) {
                // mask shape: [batch, 1, 1] for broadcasting with [batch, seq, dim]
                let mask = mask.reshape((batch_size, 1, 1))?;
                match strategy {
                    SkipLayerStrategy::AttentionSkip => {
                        // Blend attention output with original hidden states
                        // hidden_states = attn_output * mask + hidden_states * (1 - mask)
                        let one = mask.ones_like()?;
                        let inv_mask = one.sub(&mask)?;
                        attn_output
                            .broadcast_mul(&mask)?
                            .add(&hidden_states.broadcast_mul(&inv_mask)?)?
                    }
                    SkipLayerStrategy::AttentionValues => {
                        // Blend attention output with raw values (for STG)
                        // hidden_states = attn_output * mask + value_for_stg * (1 - mask)
                        let one = mask.ones_like()?;
                        let inv_mask = one.sub(&mask)?;
                        attn_output
                            .broadcast_mul(&mask)?
                            .add(&value_for_stg.broadcast_mul(&inv_mask)?)?
                    }
                    _ => attn_output, // Other strategies handled at block level
                }
            } else {
                attn_output
            };

        // Output projection
        self.to_out.forward(&hidden_states_out)
    }
}

// ===========================================================================
// Timestep Embedding
// ===========================================================================

/// Sinusoidal timestep embeddings
pub struct TimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
    dim: usize,
}

impl TimestepEmbedding {
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let linear1 = linear(in_channels, out_channels, vb.pp("linear_1"))?;
        let linear2 = linear(out_channels, out_channels, vb.pp("linear_2"))?;
        Ok(Self {
            linear1,
            linear2,
            dim: in_channels,
        })
    }

    fn get_sinusoidal_embeddings(&self, timesteps: &Tensor, device: &Device) -> Result<Tensor> {
        let dtype = timesteps.dtype();
        let half_dim = self.dim / 2;

        // Compute sinusoidal embeddings natively in target dtype using affine
        let exponent = Tensor::arange(0u32, half_dim as u32, device)?
            .to_dtype(dtype)?
            .affine(1.0 / (half_dim as f64), 0.0)?
            .affine(-(10000.0f64.ln()), 0.0)?
            .exp()?;

        let emb = timesteps
            .unsqueeze(1)?
            .broadcast_mul(&exponent.unsqueeze(0)?)?;

        let sin_emb = emb.sin()?;
        let cos_emb = emb.cos()?;

        // diffusers uses flip_sin_to_cos=True, so cos comes first
        Tensor::cat(&[cos_emb, sin_emb], D::Minus1)
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let device = timesteps.device();
        let emb = self.get_sinusoidal_embeddings(timesteps, device)?;
        let emb = self.linear1.forward(&emb)?.silu()?;
        self.linear2.forward(&emb)
    }
}

// ===========================================================================
// AdaLayerNormSingle
// ===========================================================================

/// Adaptive Layer Normalization for single scale-shift conditioning
/// Used in PixArt-α style transformers
pub struct AdaLayerNormSingle {
    emb: TimestepEmbedding,
    linear: Linear,
    dim: usize,
}

impl AdaLayerNormSingle {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        // Timestep embedding uses dim as input (sinusoidal) and dim as output
        let emb = TimestepEmbedding::new(vb.pp("emb.timestep_embedder"), 256, dim)?;

        // Linear projects to 6 * dim for (shift1, scale1, gate1, shift2, scale2, gate2)
        let linear = linear(dim, 6 * dim, vb.pp("linear"))?;

        Ok(Self { emb, linear, dim })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        // Get timestep embedding
        let embedded = self.emb.forward(timestep)?;

        // Project to conditioning values
        let emb = embedded.silu()?;
        let out = self.linear.forward(&emb)?;

        // Reshape to (batch, 1, 6*dim) for broadcasting
        let batch = timestep.dim(0)?;
        let out = out.reshape((batch, 1, 6 * self.dim))?;

        Ok((out, embedded))
    }
}

// ===========================================================================
// CaptionProjection
// ===========================================================================

/// Projects caption embeddings (from T5) to model dimension
/// Matches diffusers PixArtAlphaTextProjection: linear_1 -> GELU -> linear_2
pub struct CaptionProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl CaptionProjection {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        let linear_1 = linear(
            config.caption_channels,
            config.hidden_size,
            vb.pp("linear_1"),
        )?;
        let linear_2 = linear(config.hidden_size, config.hidden_size, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, caption: &Tensor) -> Result<Tensor> {
        // linear_1 -> GELU(tanh) -> linear_2
        let hidden = self.linear_1.forward(caption)?;
        let hidden = hidden.gelu()?; // tanh approximation to match diffusers
        self.linear_2.forward(&hidden)
    }
}

// ===========================================================================
// BasicTransformerBlock
// ===========================================================================

/// A basic transformer block with AdaLN-Zero modulation (LTX-Video style)
/// Matches diffusers LTXVideoTransformerBlock with non-affine norm1/norm2
///
/// Structure:
/// - norm1 + self-attention (attn1) with RoPE
/// - norm2 + cross-attention (attn2)
/// - norm (via AdaLN) + feedforward (ff) with GEGLU
/// - AdaLN modulation via scale_shift_table
pub struct BasicTransformerBlock {
    norm1: RMSNormNoWeight, // Pre self-attention normalization (NO weights, elementwise_affine=False)
    norm2: RMSNormNoWeight, // Pre cross-attention normalization (NO weights)
    attn1: Attention,       // Self-attention (has internal QK-norm)
    attn2: Option<Attention>, // Cross-attention (optional)
    ff: FeedForward,
    scale_shift_table: Tensor, // (6, dim) for AdaLN modulation
    hidden_size: usize,
}

impl BasicTransformerBlock {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        // Pre-normalization layers (RMSNorm with NO weights - elementwise_affine=False)
        // LTX-Video uses non-affine RMSNorm for norm1/norm2
        let norm1 = RMSNormNoWeight::new(1e-6);
        let norm2 = RMSNormNoWeight::new(1e-6);

        // Self-attention (no cross_attention_dim) - has internal QK-norm
        let attn1 = Attention::new(vb.pp("attn1"), config, None)?;

        // Cross-attention with caption (uses caption_channels projected to hidden_size)
        let attn2 = if config.caption_channels > 0 {
            Some(Attention::new(
                vb.pp("attn2"),
                config,
                Some(config.hidden_size), // Already projected
            )?)
        } else {
            None
        };

        let ff = FeedForward::new(vb.pp("ff"), config)?;

        // Scale-shift table for AdaLN-Zero: 6 params (shift, scale, gate for attn and mlp)
        let scale_shift_table = vb.get_with_hints(
            (6, config.hidden_size),
            "scale_shift_table",
            candle_nn::init::Init::Randn {
                mean: 0.0,
                stdev: (config.hidden_size as f64).powf(-0.5),
            },
        )?;

        Ok(Self {
            norm1,
            norm2,
            attn1,
            attn2,
            ff,
            scale_shift_table,
            hidden_size: config.hidden_size,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        freqs_cis: Option<&(Tensor, Tensor)>,
        encoder_hidden_states: Option<&Tensor>,
        timestep: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
        skip_layer_strategy: Option<SkipLayerStrategy>,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        let device = hidden_states.device();

        // Store original hidden states for TransformerBlock skip strategy
        let original_hidden_states = hidden_states.clone();

        // Parse AdaLN conditioning from timestep
        // timestep comes as (batch, 1, 6 * hidden_size) from AdaLayerNormSingle
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            if let Some(ts) = timestep {
                let dim = self.hidden_size;

                // ts shape: (batch, 1, 6 * dim) -> squeeze to (batch, 6 * dim)
                let ts_squeezed = ts.squeeze(1)?;

                // Add scale_shift_table: (6, dim)
                let table_expanded = self.scale_shift_table.unsqueeze(0)?; // (1, 6, dim)

                // Reshape ts to (batch, 6, dim)
                let ts_reshaped = ts_squeezed.reshape((batch_size, 6, dim))?;
                let ada_values = ts_reshaped.broadcast_add(&table_expanded)?; // (batch, 6, dim)

                // Split into 6 components, each (batch, 1, dim) for broadcasting with (batch, N, dim)
                let shift_msa = ada_values.i((.., 0..1, ..))?; // (batch, 1, dim)
                let scale_msa = ada_values.i((.., 1..2, ..))?;
                let gate_msa = ada_values.i((.., 2..3, ..))?;
                let shift_mlp = ada_values.i((.., 3..4, ..))?;
                let scale_mlp = ada_values.i((.., 4..5, ..))?;
                let gate_mlp = ada_values.i((.., 5..6, ..))?;

                (
                    Some(shift_msa),
                    Some(scale_msa),
                    Some(gate_msa),
                    Some(shift_mlp),
                    Some(scale_mlp),
                    Some(gate_mlp),
                )
            } else {
                (None, None, None, None, None, None)
            };

        // 1. Self-Attention with AdaLN-Zero
        // Apply learned normalization (matches Python norm1)
        let mut attn_input = self.norm1.forward(hidden_states)?;

        // Apply AdaLN modulation: norm(x) * (1 + scale) + shift
        if let (Some(scale), Some(shift)) = (&scale_msa, &shift_msa) {
            let one = Tensor::ones(scale.shape(), scale.dtype(), device)?;
            attn_input = attn_input
                .broadcast_mul(&one.add(scale)?)?
                .broadcast_add(shift)?;
        }

        let attn_output = self.attn1.forward(
            &attn_input,
            freqs_cis,
            None,
            None,
            skip_layer_mask,
            skip_layer_strategy,
        )?;

        // Apply gate
        let attn_output = if let Some(gate) = &gate_msa {
            attn_output.broadcast_mul(gate)?
        } else {
            attn_output
        };

        let mut hidden_states = hidden_states.add(&attn_output)?;

        // 2. Cross-Attention (if encoder_hidden_states provided)
        // Note: Cross-attention does NOT use RoPE (image_rotary_emb=None in Python)
        if let (Some(attn2), Some(enc_hidden)) = (&self.attn2, encoder_hidden_states) {
            let cross_attn_output = attn2.forward(
                &hidden_states,
                None, // No RoPE for cross-attention!
                Some(enc_hidden),
                encoder_attention_mask,
                None, // No skip layer mask for cross-attention
                None,
            )?;
            hidden_states = hidden_states.add(&cross_attn_output)?;
        }

        // 3. Feed-Forward with AdaLN-Zero
        // Apply learned normalization (matches Python norm2)
        let mut ff_input = self.norm2.forward(&hidden_states)?;

        // Apply AdaLN modulation
        if let (Some(scale), Some(shift)) = (&scale_mlp, &shift_mlp) {
            let one = Tensor::ones(scale.shape(), scale.dtype(), device)?;
            ff_input = ff_input
                .broadcast_mul(&one.add(scale)?)?
                .broadcast_add(shift)?;
        }

        let ff_output = self.ff.forward(&ff_input)?;

        // Apply gate
        let ff_output = if let Some(gate) = &gate_mlp {
            ff_output.broadcast_mul(gate)?
        } else {
            ff_output
        };

        let hidden_states = hidden_states.add(&ff_output)?;

        // 4. Apply TransformerBlock skip strategy (entire block skip)
        // Matches Python attention.py lines 312-319
        if let (Some(mask), Some(SkipLayerStrategy::TransformerBlock)) =
            (skip_layer_mask, skip_layer_strategy)
        {
            // mask shape: [batch] -> [batch, 1, 1] for broadcasting
            let mask = mask.reshape((batch_size, 1, 1))?;
            let one = mask.ones_like()?;
            let inv_mask = one.sub(&mask)?;
            // Blend block output with original input
            hidden_states
                .broadcast_mul(&mask)?
                .add(&original_hidden_states.broadcast_mul(&inv_mask)?)
        } else {
            Ok(hidden_states)
        }
    }
}

// ===========================================================================
// Transformer3DModel
// ===========================================================================

/// Full 3D Transformer model for video diffusion (LTX-Video backbone)
pub struct Transformer3DModel {
    patchify_proj: Linear,
    transformer_blocks: Vec<BasicTransformerBlock>,
    scale_shift_table: Tensor,
    proj_out: Linear,
    adaln_single: AdaLayerNormSingle,
    caption_projection: Option<CaptionProjection>,
    rope: FractionalRoPE,
    #[allow(dead_code)]
    config: DitConfig,
}

impl Transformer3DModel {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        let inner_dim = config.num_heads * (config.hidden_size / config.num_heads);

        // Patchify projection
        let patchify_proj = linear(config.in_channels, inner_dim, vb.pp("patchify_proj"))?;

        // Transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let block =
                BasicTransformerBlock::new(vb.pp(format!("transformer_blocks.{}", i)), config)?;
            transformer_blocks.push(block);
        }

        // Output scale_shift_table
        let scale_shift_table = vb.get_with_hints(
            (2, inner_dim),
            "scale_shift_table",
            candle_nn::init::Init::Randn {
                mean: 0.0,
                stdev: (inner_dim as f64).powf(-0.5),
            },
        )?;

        let proj_out = linear(inner_dim, config.in_channels, vb.pp("proj_out"))?;

        // AdaLN single
        let adaln_single = AdaLayerNormSingle::new(vb.pp("adaln_single"), inner_dim)?;

        // Caption projection
        let caption_projection = if config.caption_channels > 0 {
            Some(CaptionProjection::new(vb.pp("caption_projection"), config)?)
        } else {
            None
        };

        // RoPE - use hidden_size (not head_dim!) to match diffusers LTXVideoRotaryPosEmbed
        // Diffusers applies RoPE on full inner_dim before splitting into heads
        let rope = FractionalRoPE::new(config.hidden_size, 10000.0, 4096);

        Ok(Self {
            patchify_proj,
            transformer_blocks,
            scale_shift_table,
            proj_out,
            adaln_single,
            caption_projection,
            rope,
            config: config.clone(),
        })
    }

    /// Precompute RoPE frequencies from indices grid
    fn precompute_freqs_cis(
        &self,
        indices_grid: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.rope.compute_freqs_cis(indices_grid, device)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        indices_grid: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        timestep: &Tensor,
        _attention_mask: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        skip_layer_mask: Option<&[Tensor]>,
        skip_layer_strategy: Option<SkipLayerStrategy>,
    ) -> Result<Tensor> {
        let device = hidden_states.device();

        // Get input dimensions - hidden_states is 5D: (B, C, T, H, W)
        let (batch_size, _c, t, h, w) = hidden_states.dims5()?;
        let num_patches = t * h * w;

        // 1. Reshape 5D -> 3D: (B, C, T, H, W) -> (B, T*H*W, C)
        let hidden_states = hidden_states
            .permute((0, 2, 3, 4, 1))? // (B, T, H, W, C)
            .reshape((batch_size, num_patches, self.config.in_channels))?;

        // 2. Input projection: (B, N, in_channels) -> (B, N, hidden_size)
        let mut hidden_states = self.patchify_proj.forward(&hidden_states)?;

        // 3. Compute RoPE frequencies (convert indices_grid to model dtype)
        let indices_grid = indices_grid.to_dtype(hidden_states.dtype())?;
        let freqs_cis = self.precompute_freqs_cis(&indices_grid, device)?;

        // 4. Apply timestep scale multiplier (matches Python transformer3d.py line 418-419)
        let timestep = if let Some(scale) = self.config.timestep_scale_multiplier {
            timestep.affine(scale, 0.0)?
        } else {
            timestep.clone()
        };

        // 5. Timestep conditioning
        let (timestep_emb, embedded_timestep) = self.adaln_single.forward(&timestep)?;

        // 5. Caption projection
        let encoder_hidden_states =
            if let (Some(proj), Some(enc)) = (&self.caption_projection, encoder_hidden_states) {
                Some(proj.forward(enc)?)
            } else {
                None
            };

        // 5b. Convert encoder attention mask to additive bias: (1 - mask) * -10000.0
        // This makes masked positions have very negative attention scores
        let encoder_attention_bias = if let Some(mask) = encoder_attention_mask {
            let dtype = hidden_states.dtype();
            let ones = mask.ones_like()?;
            // Use affine for -10000 multiplication to preserve dtype (BF16)
            let bias = ones.sub(mask)?.to_dtype(dtype)?.affine(-10000.0, 0.0)?;
            // Expand for multi-head attention: [B, seq] -> [B, 1, 1, seq]
            Some(bias.unsqueeze(1)?.unsqueeze(2)?)
        } else {
            None
        };

        // 6. Transformer blocks
        for (block_idx, block) in self.transformer_blocks.iter().enumerate() {
            // Get per-block skip layer mask if provided
            let block_skip_mask = skip_layer_mask.and_then(|masks| masks.get(block_idx));
            hidden_states = block.forward(
                &hidden_states,
                Some(&freqs_cis),
                encoder_hidden_states.as_ref(),
                Some(&timestep_emb),
                encoder_attention_bias.as_ref(),
                block_skip_mask,
                skip_layer_strategy,
            )?;
        }

        // 6. Output normalization and modulation

        // Apply LayerNorm first (without learnable weights)
        // LayerNorm: (x - mean) / sqrt(var + eps)
        let layer_norm_out = |x: &Tensor| -> Result<Tensor> {
            let eps = 1e-6f32;
            let _dtype = x.dtype();
            let mean = x.mean_keepdim(D::Minus1)?;
            let x_centered = x.broadcast_sub(&mean)?;
            let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
            // Use affine to add eps while preserving dtype (BF16)
            let std = var.affine(1.0, eps as f64)?.sqrt()?;
            x_centered.broadcast_div(&std)
        };
        let hidden_states = layer_norm_out(&hidden_states)?;

        // Apply final scale-shift
        // embedded_timestep: (batch, dim) -> (batch, 1, dim)
        let embedded_timestep = embedded_timestep.unsqueeze(1)?;
        // scale_shift_table: (2, dim) -> (1, 2, dim)
        let scale_shift = self
            .scale_shift_table
            .unsqueeze(0)?
            .broadcast_add(&embedded_timestep)?; // (batch, 2, dim)

        // Extract shift and scale, keep middle dim for broadcasting: (batch, 1, dim)
        let shift = scale_shift.i((.., 0..1, ..))?;
        let scale = scale_shift.i((.., 1..2, ..))?;

        // hidden_states: (batch, N, dim), scale/shift: (batch, 1, dim)
        let one = Tensor::ones(scale.shape(), scale.dtype(), device)?;
        let hidden_states = hidden_states
            .broadcast_mul(&one.add(&scale)?)?
            .broadcast_add(&shift)?;

        // Final projection: (B, N, hidden_size) -> (B, N, in_channels)
        let hidden_states = self.proj_out.forward(&hidden_states)?;

        // Reshape back to 5D: (B, N, C) -> (B, C, T, H, W)
        let hidden_states = hidden_states
            .reshape((batch_size, t, h, w, self.config.in_channels))?
            .permute((0, 4, 1, 2, 3))?;

        Ok(hidden_states)
    }

    /// Create skip layer mask for STG (Spatio-Temporal Guidance)
    ///
    /// Matches Python transformer3d.py create_skip_layer_mask (lines 173-203)
    ///
    /// Creates a per-block mask that determines which batch elements get
    /// the skip layer treatment. Used for STG where perturbed samples
    /// need different processing than conditioned/unconditioned.
    ///
    /// # Arguments
    /// * `batch_size` - Size of a single condition batch
    /// * `num_conds` - Total number of conditions (e.g., 3 for CFG+STG: uncond, cond, perturbed)
    /// * `perturbed_index` - Which condition index is the perturbed one (usually num_conds - 1)
    /// * `skip_blocks` - List of block indices where skip should be applied
    /// * `device` - Device for tensor creation
    ///
    /// # Returns
    /// Vec of Tensors, one per transformer block. Each tensor has shape [batch_size * num_conds]
    /// where perturbed samples have value 0.0 (skip) in specified blocks, others have 1.0 (keep).
    pub fn create_skip_layer_mask(
        &self,
        batch_size: usize,
        num_conds: usize,
        perturbed_index: usize,
        skip_blocks: &[usize],
        device: &Device,
    ) -> Result<Vec<Tensor>> {
        let total_batch = batch_size * num_conds;
        let num_blocks = self.transformer_blocks.len();

        // Create masks for each block
        let mut masks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            // Check if this block should apply skip
            let should_skip = skip_blocks.contains(&block_idx);

            // Create mask: 1.0 for keep, 0.0 for skip
            // Only the perturbed samples get skip treatment
            let mut mask_data = vec![1.0f32; total_batch];

            if should_skip {
                // Mark perturbed samples as 0.0 (skip)
                // perturbed samples are at indices: perturbed_index * batch_size .. (perturbed_index + 1) * batch_size
                let start = perturbed_index * batch_size;
                let end = start + batch_size;
                for item in mask_data.iter_mut().take(end).skip(start) {
                    *item = 0.0;
                }
            }

            let mask = Tensor::from_vec(mask_data, total_batch, device)?;
            masks.push(mask);
        }

        Ok(masks)
    }

    /// Get the number of transformer blocks
    pub fn num_blocks(&self) -> usize {
        self.transformer_blocks.len()
    }
}
// ===========================================================================
// PatchEmbedding - Improved 5D tensor handling with separate T/HW patch sizes
// ===========================================================================

/// Spatial-temporal patch embedding for 5D video tensors
/// Converts (B, C, T, H, W) to (B, num_patches, hidden_size)
///
/// Supports:
/// - Separate temporal (patch_size_t) and spatial (patch_size) patch sizes
/// - 5D tensor patchification with efficient reshaping
/// - Indices grid generation for RoPE positional embeddings
/// - Unpatchify for inverse transformation (decoding)
///
/// # Example
/// ```ignore
/// let config = DitConfig {
///     patch_size: 2,      // Spatial: H/2, W/2
///     patch_size_t: Some(1), // Temporal: T/1 (no temporal patching)
///     in_channels: 128,
///     hidden_size: 1152,
///     ..Default::default()
/// };
/// let embed = PatchEmbedding::new(vb, &config)?;
/// let x = Tensor::randn(0f32, 1.0, (1, 128, 8, 32, 32), &device)?;
/// let patches = embed.forward(&x)?;  // (1, 8*16*16, 1152)
/// ```
pub struct PatchEmbedding {
    proj: Linear,
    /// Spatial patch size (applies to H, W)
    patch_size: usize,
    /// Temporal patch size (applies to T)
    patch_size_t: usize,
    /// Input channels (for unpatchify)
    in_channels: usize,
    /// Hidden size (for future inverse projection)
    #[allow(dead_code)]
    hidden_size: usize,
}

impl PatchEmbedding {
    /// Create a new PatchEmbedding
    ///
    /// # Arguments
    /// * `vb` - VarBuilder for weight initialization
    /// * `config` - DiT configuration with patch sizes
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        let patch_size = config.patch_size;
        let patch_size_t = config.patch_size_t.unwrap_or(patch_size);

        // Input dimension: C * patch_size_t * patch_size * patch_size
        let in_dim = config.in_channels * patch_size_t * patch_size * patch_size;
        let proj = linear(in_dim, config.hidden_size, vb.pp("proj"))?;

        Ok(Self {
            proj,
            patch_size,
            patch_size_t,
            in_channels: config.in_channels,
            hidden_size: config.hidden_size,
        })
    }

    /// Get the spatial patch size
    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Get the temporal patch size
    pub fn patch_size_t(&self) -> usize {
        self.patch_size_t
    }

    /// Patchify a 5D video tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (B, C, T, H, W)
    ///
    /// # Returns
    /// Tensor of shape (B, num_patches, hidden_size)
    /// where num_patches = (T/patch_size_t) * (H/patch_size) * (W/patch_size)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let ps = self.patch_size;
        let ps_t = self.patch_size_t;

        // Validate dimensions
        if t % ps_t != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Temporal dimension {} is not divisible by patch_size_t {}",
                t, ps_t
            )));
        }
        if h % ps != 0 || w % ps != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Spatial dimensions ({}, {}) are not divisible by patch_size {}",
                h, w, ps
            )));
        }

        // Fast path: no patchification needed
        if ps == 1 && ps_t == 1 {
            let x = x.permute((0, 2, 3, 4, 1))?; // (B, T, H, W, C)
            let x = x.reshape((b, t * h * w, c))?; // (B, T*H*W, C)
            return self.proj.forward(&x);
        }

        // Calculate patch dimensions
        let t_patches = t / ps_t;
        let h_patches = h / ps;
        let w_patches = w / ps;
        let num_patches = t_patches * h_patches * w_patches;
        let patch_dim = c * ps_t * ps * ps;

        // Patchify using multi-step approach (Candle supports up to 6D)
        let x =
            self.patchify_internal(x, b, c, t, h, w, t_patches, h_patches, w_patches, ps_t, ps)?;

        // x is now (B, num_patches, patch_dim)
        debug_assert_eq!(x.dims(), [b, num_patches, patch_dim]);

        self.proj.forward(&x)
    }

    /// Internal patchification logic
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn patchify_internal(
        &self,
        x: &Tensor,
        b: usize,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        t_patches: usize,
        h_patches: usize,
        w_patches: usize,
        ps_t: usize,
        ps: usize,
    ) -> Result<Tensor> {
        let num_patches = t_patches * h_patches * w_patches;
        let patch_dim = c * ps_t * ps * ps;

        // Handle different patch size combinations
        if ps_t == 1 {
            // Only spatial patching (common case for LTX-Video)
            // (B, C, T, H, W) -> (B*T, C, H, W)
            let x = x.reshape((b * t, c, h, w))?;

            // (B*T, C, H, W) -> (B*T, C, H/ps, ps, W/ps, ps)
            let x = x.reshape((b * t, c, h_patches, ps, w_patches, ps))?;

            // Permute: (B*T, H/ps, W/ps, C, ps, ps)
            let x = x.permute((0, 2, 4, 1, 3, 5))?;

            // Flatten to patches: (B*T, H'*W', C*ps*ps)
            let x = x.reshape((b * t, h_patches * w_patches, c * ps * ps))?;

            // Reshape back to batch: (B, T*H'*W', C*ps*ps)
            x.reshape((b, num_patches, patch_dim))
        } else if ps == 1 {
            // Only temporal patching
            // (B, C, T, H, W) -> (B, C, T/ps_t, ps_t, H*W)
            let x = x.reshape((b, c, t_patches, ps_t, h * w))?;

            // (B, C, T/ps_t, ps_t, H*W) -> (B, T/ps_t, C*ps_t, H*W)
            let x = x.permute((0, 2, 1, 3, 4))?;
            let x = x.reshape((b, t_patches, c * ps_t, h * w))?;

            // (B, T/ps_t, C*ps_t, H*W) -> (B, T'*H*W, C*ps_t)
            let x = x.permute((0, 1, 3, 2))?;
            x.reshape((b, num_patches, patch_dim))
        } else {
            // Full spatial-temporal patching
            // Step 1: Handle temporal dimension
            // (B, C, T, H, W) -> (B, C, T/ps_t, ps_t, H, W)
            let x = x.reshape((b, c, t_patches, ps_t, h * w))?;

            // (B, C, T/ps_t, ps_t, H*W) -> (B, T/ps_t, C, ps_t, H*W)
            let x = x.permute((0, 2, 1, 3, 4))?;

            // Flatten temporal: (B*T', C*ps_t, H, W)
            let x = x.reshape((b * t_patches, c * ps_t, h, w))?;

            // Step 2: Handle spatial dimensions
            // (B*T', C*ps_t, H, W) -> (B*T', C*ps_t, H/ps, ps, W/ps, ps)
            let x = x.reshape((b * t_patches, c * ps_t, h_patches, ps, w_patches, ps))?;

            // Permute: (B*T', H/ps, W/ps, C*ps_t, ps, ps)
            let x = x.permute((0, 2, 4, 1, 3, 5))?;

            // Flatten to patches: (B*T', H'*W', C*ps_t*ps*ps)
            let x = x.reshape((b * t_patches, h_patches * w_patches, patch_dim))?;

            // Step 3: Combine batch and temporal patch dimensions
            // (B*T', H'*W', patch_dim) -> (B, T'*H'*W', patch_dim)
            x.reshape((b, num_patches, patch_dim))
        }
    }

    /// Unpatchify: Convert patches back to 5D video tensor
    ///
    /// # Arguments
    /// * `patches` - Tensor of shape (B, num_patches, hidden_size) or (B, num_patches, patch_dim)
    /// * `t_patches` - Number of temporal patches
    /// * `h_patches` - Number of height patches
    /// * `w_patches` - Number of width patches
    ///
    /// # Returns
    /// Tensor of shape (B, C, T, H, W)
    pub fn unpatchify(
        &self,
        patches: &Tensor,
        t_patches: usize,
        h_patches: usize,
        w_patches: usize,
    ) -> Result<Tensor> {
        let (b, num_patches, dim) = patches.dims3()?;
        let ps = self.patch_size;
        let ps_t = self.patch_size_t;
        let c = self.in_channels;

        let expected_patches = t_patches * h_patches * w_patches;
        if num_patches != expected_patches {
            return Err(candle_core::Error::Msg(format!(
                "Number of patches {} doesn't match expected {} ({}*{}*{})",
                num_patches, expected_patches, t_patches, h_patches, w_patches
            )));
        }

        let patch_dim = c * ps_t * ps * ps;

        // If dim is hidden_size, we need inverse projection (not implemented here)
        // This method assumes patches are already in patch_dim format
        if dim != patch_dim {
            return Err(candle_core::Error::Msg(format!(
                "Unpatchify expects patch_dim {} but got {}. Use proj_out first.",
                patch_dim, dim
            )));
        }

        // Reverse the patchification process
        let t = t_patches * ps_t;
        let h = h_patches * ps;
        let w = w_patches * ps;

        // Fast path
        if ps == 1 && ps_t == 1 {
            let x = patches.reshape((b, t, h, w, c))?;
            return x.permute((0, 4, 1, 2, 3)); // (B, C, T, H, W)
        }

        // Handle spatial-only patching (most common)
        if ps_t == 1 {
            // (B, T*H'*W', C*ps*ps) -> (B*T, H'*W', C*ps*ps)
            let x = patches.reshape((b * t, h_patches * w_patches, c * ps * ps))?;

            // (B*T, H'*W', C*ps*ps) -> (B*T, H', W', C, ps, ps)
            let x = x.reshape((b * t, h_patches, w_patches, c, ps, ps))?;

            // Permute: (B*T, C, H', ps, W', ps)
            let x = x.permute((0, 3, 1, 4, 2, 5))?;

            // Reshape to spatial: (B*T, C, H, W)
            let x = x.reshape((b * t, c, h, w))?;

            // Reshape back to batch: (B, C, T, H, W)
            return x.reshape((b, c, t, h, w));
        }

        // Full unpatchification
        // (B, T'*H'*W', patch_dim) -> (B*T', H'*W', patch_dim)
        let x = patches.reshape((b * t_patches, h_patches * w_patches, patch_dim))?;

        // (B*T', H'*W', C*ps_t*ps*ps) -> (B*T', H', W', C*ps_t, ps, ps)
        let x = x.reshape((b * t_patches, h_patches, w_patches, c * ps_t, ps, ps))?;

        // Permute: (B*T', C*ps_t, H', ps, W', ps)
        let x = x.permute((0, 3, 1, 4, 2, 5))?;

        // Reshape to spatial: (B*T', C*ps_t, H, W)
        let x = x.reshape((b * t_patches, c * ps_t, h, w))?;

        // (B*T', C*ps_t, H, W) -> (B, T', C*ps_t, H*W)
        let x = x.reshape((b, t_patches, c * ps_t, h * w))?;

        // (B, T', C*ps_t, H*W) -> (B, C, T', ps_t, H*W)
        let x = x.permute((0, 2, 1, 3))?;
        let x = x.reshape((b, c, t_patches, ps_t, h * w))?;

        // (B, C, T, H*W) -> (B, C, T, H, W)
        let x = x.reshape((b, c, t, h * w))?;
        x.reshape((b, c, t, h, w))
    }

    /// Generate indices grid for RoPE from spatial dimensions
    ///
    /// Creates normalized [0, 1] coordinates for each patch position
    /// in (T, H, W) order.
    ///
    /// # Arguments
    /// * `batch_size` - Batch size
    /// * `t` - Original temporal dimension (before patching)
    /// * `h` - Original height dimension (before patching)
    /// * `w` - Original width dimension (before patching)
    /// * `device` - Device for tensor creation
    ///
    /// # Returns
    /// Tensor of shape (batch_size, 3, num_patches)
    pub fn get_indices_grid(
        &self,
        batch_size: usize,
        t: usize,
        h: usize,
        w: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let ps = self.patch_size;
        let ps_t = self.patch_size_t;

        let t_patches = t / ps_t;
        let h_patches = h / ps;
        let w_patches = w / ps;
        let num_patches = t_patches * h_patches * w_patches;

        let mut indices = Vec::with_capacity(batch_size * 3 * num_patches);

        // T coordinates (normalized to [0, 1])
        for _b in 0..batch_size {
            for ti in 0..t_patches {
                let t_coord = if t_patches > 1 {
                    ti as f32 / (t_patches - 1) as f32
                } else {
                    0.0
                };
                for _hi in 0..h_patches {
                    for _wi in 0..w_patches {
                        indices.push(t_coord);
                    }
                }
            }
        }

        // H coordinates (normalized to [0, 1])
        for _b in 0..batch_size {
            for _ti in 0..t_patches {
                for hi in 0..h_patches {
                    let h_coord = if h_patches > 1 {
                        hi as f32 / (h_patches - 1) as f32
                    } else {
                        0.0
                    };
                    for _wi in 0..w_patches {
                        indices.push(h_coord);
                    }
                }
            }
        }

        // W coordinates (normalized to [0, 1])
        for _b in 0..batch_size {
            for _ti in 0..t_patches {
                for _hi in 0..h_patches {
                    for wi in 0..w_patches {
                        let w_coord = if w_patches > 1 {
                            wi as f32 / (w_patches - 1) as f32
                        } else {
                            0.0
                        };
                        indices.push(w_coord);
                    }
                }
            }
        }

        Tensor::from_vec(indices, (batch_size, 3, num_patches), device)
    }

    /// Calculate the number of patches for given dimensions
    ///
    /// # Arguments
    /// * `t` - Temporal dimension
    /// * `h` - Height dimension
    /// * `w` - Width dimension
    ///
    /// # Returns
    /// Number of patches
    pub fn num_patches(&self, t: usize, h: usize, w: usize) -> usize {
        (t / self.patch_size_t) * (h / self.patch_size) * (w / self.patch_size)
    }

    /// Get patch dimensions after patchification
    ///
    /// # Arguments
    /// * `t` - Original temporal dimension
    /// * `h` - Original height dimension
    /// * `w` - Original width dimension
    ///
    /// # Returns
    /// Tuple of (t_patches, h_patches, w_patches)
    pub fn patch_dims(&self, t: usize, h: usize, w: usize) -> (usize, usize, usize) {
        (
            t / self.patch_size_t,
            h / self.patch_size,
            w / self.patch_size,
        )
    }
}

// ===========================================================================
// JointAttention - Legacy API compatibility
// ===========================================================================

/// Joint spatiotemporal attention (legacy API for backward compatibility)
pub struct JointAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl JointAttention {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("proj"))?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &FractionalRoPE,
        t_coords: &Tensor,
        h_coords: &Tensor,
        w_coords: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _d) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);

        // Compute QKV
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?.contiguous()?; // (3, B, H, L, D)

        let q = qkv.get(0)?.contiguous()?; // (B, H, L, D)
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;

        // Apply RoPE
        let (q, k) = rope.apply_3d(&q, &k, t_coords, h_coords, w_coords)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;

        // Compute attention scores
        let k_t = k.t()?.contiguous()?;
        let scores = q.matmul(&k_t)?;
        // Use affine for attention scale to preserve dtype (BF16)
        let scores = scores.affine(self.scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Apply attention to values
        let out = attn.matmul(&v)?; // (B, H, L, D)
        let out = out.permute((0, 2, 1, 3))?.contiguous()?; // (B, L, H, D)
        let out = out.flatten_from(2)?; // (B, L, H*D)

        // Project
        self.proj.forward(&out)
    }
}

// ===========================================================================
// ModulationLayer - Legacy API
// ===========================================================================

/// Modulation layer for adaptive normalization (legacy API)
pub struct ModulationLayer {
    linear: Linear,
}

impl ModulationLayer {
    pub fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        // Combines t_emb (256) and c_emb (4096) -> hidden_size
        let linear = linear_no_bias(4096 + 256, hidden_size, vb)?;
        Ok(Self { linear })
    }

    pub fn forward(&self, t_emb: &Tensor, c_emb: &Tensor) -> Result<Tensor> {
        let combined = Tensor::cat(&[c_emb, t_emb], D::Minus1)?;
        let out = self.linear.forward(&combined)?;
        // Sigmoid for scale in [0, 1], unsqueeze for broadcasting with (batch, seq, hidden)
        let out = candle_nn::Activation::Sigmoid.forward(&out)?;
        out.unsqueeze(1) // (batch, 1, hidden_size) for broadcasting
    }
}

// ===========================================================================
// DiTBlock - Legacy API
// ===========================================================================

/// DiT block with joint spatiotemporal attention (legacy API)
pub struct DiTBlock {
    norm1: LayerNorm,
    attn: JointAttention,
    norm2: LayerNorm,
    mlp: LegacyFeedForward,
    mod1: ModulationLayer,
    mod2: ModulationLayer,
}

/// Legacy FeedForward for DiTBlock
pub struct LegacyFeedForward {
    fc1: Linear,
    fc2: Linear,
    mod_scale: ModulationLayer,
    mod_gate: ModulationLayer,
}

impl LegacyFeedForward {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        let mlp_dim = (config.hidden_size as f64 * config.mlp_ratio) as usize;
        let fc1 = linear_no_bias(config.hidden_size, mlp_dim, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(mlp_dim, config.hidden_size, vb.pp("fc2"))?;
        let mod_scale = ModulationLayer::new(vb.pp("mod_scale"), mlp_dim)?;
        let mod_gate = ModulationLayer::new(vb.pp("mod_gate"), mlp_dim)?;

        Ok(Self {
            fc1,
            fc2,
            mod_scale,
            mod_gate,
        })
    }

    pub fn forward(&self, x: &Tensor, t_emb: &Tensor, c_emb: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let scale = self.mod_scale.forward(t_emb, c_emb)?;
        let gate = self.mod_gate.forward(t_emb, c_emb)?;
        let x = x.broadcast_mul(&scale)?;
        let x = x.gelu()?;
        let x = x.broadcast_mul(&gate)?;
        self.fc2.forward(&x)
    }
}

impl DiTBlock {
    pub fn new(vb: VarBuilder, config: &DitConfig) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("norm1"))?;
        let attn = JointAttention::new(vb.pp("attn"), config)?;
        let norm2 = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("norm2"))?;
        let mlp = LegacyFeedForward::new(vb.pp("mlp"), config)?;
        let mod1 = ModulationLayer::new(vb.pp("mod1"), config.hidden_size)?;
        let mod2 = ModulationLayer::new(vb.pp("mod2"), config.hidden_size)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            mod1,
            mod2,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        c_emb: &Tensor,
        rope: &FractionalRoPE,
        t_coords: &Tensor,
        h_coords: &Tensor,
        w_coords: &Tensor,
    ) -> Result<Tensor> {
        // Adaptive layer norm with modulation
        let mod1_scale = self.mod1.forward(t_emb, c_emb)?;
        let x_norm = self.norm1.forward(x)?;
        let x_norm = x_norm.broadcast_mul(&mod1_scale)?;

        // Joint attention
        let x_attn = self
            .attn
            .forward(&x_norm, rope, t_coords, h_coords, w_coords)?;
        let x = x.add(&x_attn)?;

        // Second adaptive layer norm
        let mod2_scale = self.mod2.forward(t_emb, c_emb)?;
        let x_norm = self.norm2.forward(&x)?;
        let x_norm = x_norm.broadcast_mul(&mod2_scale)?;

        // MLP
        let x_mlp = self.mlp.forward(&x_norm, t_emb, c_emb)?;
        let x = x.add(&x_mlp)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_patch_embedding() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = DitConfig::default();
        let embed = PatchEmbedding::new(vb.pp("embed"), &config)?;
        let x = Tensor::randn(0f32, 1.0, (1, 128, 8, 16, 16), &device)?;
        let out = embed.forward(&x)?;

        // After patch_size=2: T=4, H=8, W=8 => 4*8*8 = 256 patches
        assert_eq!(out.dim(0)?, 1);
        assert_eq!(out.dim(1)?, 256);
        assert_eq!(out.dim(2)?, config.hidden_size);
        Ok(())
    }

    #[test]
    fn test_geglu_activation() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 16, 64), &device)?;
        let out = x.gelu()?;
        assert_eq!(out.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_feedforward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = DitConfig {
            hidden_size: 64,
            mlp_ratio: 4.0,
            ..Default::default()
        };
        let ff = FeedForward::new(vb, &config)?;
        let x = Tensor::randn(0f32, 1.0, (2, 16, 64), &device)?;
        let out = ff.forward(&x)?;

        assert_eq!(out.dims(), x.dims());
        Ok(())
    }
}
