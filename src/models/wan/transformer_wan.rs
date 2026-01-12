//! Wan Transformer 3D implementation using common interfaces.
//!
//! Based on diffusers transformer_wan.py with VarBuilder for weight loading.

use candle_core::{D, DType, Device, IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use thiserror::Error;

// Import from common interfaces
use crate::interfaces::activations::gelu_approximate;
use crate::interfaces::conv3d::{Conv3d, Conv3dConfig};
use crate::interfaces::embeddings::{
    PixArtAlphaTextProjection, TimestepEmbedding, get_timestep_embedding,
};
use crate::interfaces::rope::get_1d_rotary_pos_embed;

// Re-export config
pub use super::config::WanTransformer3DConfig;

#[derive(Debug, Error)]
pub enum WanError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, WanError>;

// ============================================================================
// Utility functions
// ============================================================================

fn ensure(cond: bool, msg: impl Into<String>) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(WanError::InvalidArgument(msg.into()))
    }
}

// ============================================================================
// FP32LayerNorm - matches diffusers FP32LayerNorm
// ============================================================================

/// LayerNorm that computes in FP32 for numerical stability.
/// Supports both elementwise_affine=True and elementwise_affine=False.
#[derive(Clone, Debug)]
pub struct FP32LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl FP32LayerNorm {
    /// Create FP32LayerNorm with optional affine parameters.
    pub fn new(
        dim: usize,
        eps: f64,
        elementwise_affine: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let (weight, bias) = if elementwise_affine {
            (Some(vb.get(dim, "weight")?), Some(vb.get(dim, "bias")?))
        } else {
            (None, None)
        };
        Ok(Self { weight, bias, eps })
    }

    /// Create FP32LayerNorm without affine parameters.
    pub fn new_no_affine(eps: f64) -> Self {
        Self {
            weight: None,
            bias: None,
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let last_dim = x_f32.dim(D::Minus1)?;
        let mean = (x_f32.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let xc = x_f32.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let denom = (var + self.eps)?.sqrt()?;
        let mut y = xc.broadcast_div(&denom)?;

        if let Some(w) = &self.weight {
            y = y.broadcast_mul(&w.to_dtype(DType::F32)?)?;
        }
        if let Some(b) = &self.bias {
            y = y.broadcast_add(&b.to_dtype(DType::F32)?)?;
        }
        y.to_dtype(dtype)
    }
}

impl Module for FP32LayerNorm {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward(xs)
    }
}

// ============================================================================
// RmsNorm with affine weight - matches torch.nn.RMSNorm
// ============================================================================

/// RMSNorm with learnable affine weight.
#[derive(Clone, Debug)]
pub struct RmsNormAffine {
    weight: Tensor,
    eps: f64,
}

impl RmsNormAffine {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let dim = x_f32.dim(D::Minus1)? as f64;
        let ms = x_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let y = x_f32.broadcast_div(&denom)?;
        let y = y.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        y.to_dtype(dtype)
    }
}

impl Module for RmsNormAffine {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward(xs)
    }
}

// ============================================================================
// WanAttention
// ============================================================================

/// Wan attention module with RMSNorm for Q/K normalization.
#[derive(Clone, Debug)]
pub struct WanAttention {
    pub dim: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub inner_dim: usize,
    pub kv_inner_dim: usize,

    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out_0: Linear,

    pub norm_q: RmsNormAffine,
    pub norm_k: RmsNormAffine,
    // I2V support (commented out for T2V-only)
    // pub add_k_proj: Option<Linear>,
    // pub add_v_proj: Option<Linear>,
    // pub norm_added_k: Option<RmsNormAffine>,
}

impl WanAttention {
    pub fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        eps: f64,
        cross_attention_dim_head: Option<usize>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let inner_dim = dim_head * heads;
        let kv_inner_dim = cross_attention_dim_head
            .map(|d| d * heads)
            .unwrap_or(inner_dim);

        let to_q = nn::linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = nn::linear(dim, kv_inner_dim, vb.pp("to_k"))?;
        let to_v = nn::linear(dim, kv_inner_dim, vb.pp("to_v"))?;
        let to_out_0 = nn::linear(inner_dim, dim, vb.pp("to_out.0"))?;

        let norm_q = RmsNormAffine::new(inner_dim, eps, vb.pp("norm_q"))?;
        let norm_k = RmsNormAffine::new(kv_inner_dim, eps, vb.pp("norm_k"))?;

        Ok(Self {
            dim,
            heads,
            dim_head,
            inner_dim,
            kv_inner_dim,
            to_q,
            to_k,
            to_v,
            to_out_0,
            norm_q,
            norm_k,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        _attention_mask: Option<&Tensor>,
        rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> CandleResult<Tensor> {
        let encoder_hidden_states = encoder_hidden_states.unwrap_or(hidden_states);

        // QKV projections
        let mut q = self.to_q.forward(hidden_states)?;
        let mut k = self.to_k.forward(encoder_hidden_states)?;
        let v = self.to_v.forward(encoder_hidden_states)?;

        // RMSNorm on Q and K
        q = self.norm_q.forward(&q)?;
        k = self.norm_k.forward(&k)?;

        // Reshape to heads: [b, s, h, d]
        let (b, s, _) = q.dims3()?;
        let (_, s_ctx, _) = k.dims3()?;

        let q = q.reshape((b, s, self.heads, self.dim_head))?;
        let k = k.reshape((b, s_ctx, self.heads, self.dim_head))?;
        let v = v.reshape((b, s_ctx, self.heads, self.dim_head))?;

        // Apply rotary embeddings
        let (q, k) = if let Some((cos, sin)) = rotary_emb {
            (
                apply_rotary_emb(&q, cos, sin)?,
                apply_rotary_emb(&k, cos, sin)?,
            )
        } else {
            (q, k)
        };

        // Scaled dot-product attention
        let attn_out = scaled_dot_product_attention(&q, &k, &v)?;

        // Merge heads: [b, s, inner_dim]
        let attn_out = attn_out.reshape((b, s, self.inner_dim))?;

        // Output projection
        self.to_out_0.forward(&attn_out)
    }
}

/// Scaled dot-product attention: q,k,v: [b, s, h, d]
/// Uses Flash Attention when available (feature flag), falls back to naive attention otherwise.
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
    let (_, _, _, d) = q.dims4()?;
    let scale = 1f32 / (d as f32).sqrt();

    #[cfg(feature = "flash-attn")]
    {
        // Flash Attention expects [B, seq, heads, head_dim] which matches our input shape
        // Only convert/copy if necessary to save memory
        let q_bf = if q.dtype() == DType::BF16 && q.is_contiguous() {
            q.clone()
        } else {
            q.to_dtype(DType::BF16)?.contiguous()?
        };
        let k_bf = if k.dtype() == DType::BF16 && k.is_contiguous() {
            k.clone()
        } else {
            k.to_dtype(DType::BF16)?.contiguous()?
        };
        let v_bf = if v.dtype() == DType::BF16 && v.is_contiguous() {
            v.clone()
        } else {
            v.to_dtype(DType::BF16)?.contiguous()?
        };

        let out = candle_flash_attn::flash_attn(&q_bf, &k_bf, &v_bf, scale, false)?;
        
        // Only convert back if input wasn't BF16
        if q.dtype() == DType::BF16 {
            return Ok(out);
        }
        out.to_dtype(q.dtype())
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        // Fallback: naive attention (memory-intensive for long sequences)
        // Transpose to [b, h, s, d] and make contiguous for matmul
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // scores: [b, h, s, s_k]
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * scale as f64)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?; // [b, h, s, d]

        // Transpose back to [b, s, h, d]
        out.transpose(1, 2)?.contiguous()
    }
}

/// Apply rotary embeddings to 4D tensor [b, s, h, d].
fn apply_rotary_emb(x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> CandleResult<Tensor> {
    let dtype = x.dtype();
    let (b, s, h, d) = x.dims4()?;
    ensure(d % 2 == 0, "rotary requires even head_dim")
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;

    // Split into pairs: x1 = x[..., 0::2], x2 = x[..., 1::2]
    let half = d / 2;
    let x2 = x_f32.reshape((b, s, h, half, 2))?;
    let x1 = x2.i((.., .., .., .., 0))?.contiguous()?;
    let x2 = x2.i((.., .., .., .., 1))?.contiguous()?;

    // cos/sin: [1, s, 1, d] -> split to [1, s, 1, half]
    let cos = freqs_cos.to_dtype(DType::F32)?.contiguous()?;
    let sin = freqs_sin.to_dtype(DType::F32)?.contiguous()?;

    let cos_dims = cos.dims();
    let cos2 = cos.reshape((cos_dims[0], cos_dims[1], cos_dims[2], half, 2))?;
    let sin2 = sin.reshape((cos_dims[0], cos_dims[1], cos_dims[2], half, 2))?;

    let cos_even = cos2.i((.., .., .., .., 0))?.contiguous()?;
    let sin_odd = sin2.i((.., .., .., .., 1))?.contiguous()?;

    // Rotation: out[..., 0::2] = x1 * cos - x2 * sin
    //           out[..., 1::2] = x1 * sin + x2 * cos
    let out_even = x1
        .broadcast_mul(&cos_even)?
        .broadcast_sub(&x2.broadcast_mul(&sin_odd)?)?;
    let out_odd = x1
        .broadcast_mul(&sin_odd)?
        .broadcast_add(&x2.broadcast_mul(&cos_even)?)?;

    // Interleave back
    let out_even = out_even.unsqueeze(D::Minus1)?;
    let out_odd = out_odd.unsqueeze(D::Minus1)?;
    let stacked = Tensor::cat(&[&out_even, &out_odd], 4)?;
    stacked.reshape((b, s, h, d))?.to_dtype(dtype)
}

// ============================================================================
// FeedForward - gelu-approximate style
// ============================================================================

/// Feed-forward network with GELU approximate activation.
#[derive(Clone, Debug)]
pub struct WanFeedForward {
    net_0: Linear,
    net_2: Linear,
}

impl WanFeedForward {
    pub fn new(dim: usize, inner_dim: usize, vb: VarBuilder) -> CandleResult<Self> {
        // Diffusers format: ffn.net.0.proj and ffn.net.2
        let net_0 = nn::linear(dim, inner_dim, vb.pp("net.0.proj"))?;
        let net_2 = nn::linear(inner_dim, dim, vb.pp("net.2"))?;
        Ok(Self { net_0, net_2 })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = self.net_0.forward(x)?;
        let h = gelu_approximate(&h)?;
        self.net_2.forward(&h)
    }
}

impl Module for WanFeedForward {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward(xs)
    }
}

// ============================================================================
// WanRotaryPosEmbed
// ============================================================================

/// Rotary position embeddings for 3D video patches.
#[derive(Clone, Debug)]
pub struct WanRotaryPosEmbed {
    attention_head_dim: usize,
    patch_size: (usize, usize, usize),
    #[allow(dead_code)]
    max_seq_len: usize,
    #[allow(dead_code)]
    theta: f64,

    freqs_cos: Tensor,
    freqs_sin: Tensor,
    t_dim: usize,
    h_dim: usize,
    w_dim: usize,
}

impl WanRotaryPosEmbed {
    pub fn new(
        attention_head_dim: usize,
        patch_size: (usize, usize, usize),
        max_seq_len: usize,
        theta: f64,
        device: &Device,
    ) -> CandleResult<Self> {
        // Dimension split: h_dim = w_dim = 2*(head_dim//6), t_dim = head_dim - h_dim - w_dim
        let hw = 2 * (attention_head_dim / 6);
        let t_dim = attention_head_dim - 2 * hw;
        let h_dim = hw;
        let w_dim = hw;

        // Generate 1D rotary embeddings for each axis
        let (cos_t, sin_t) = get_1d_rotary_pos_embed(t_dim, max_seq_len, theta, device)?;
        let (cos_h, sin_h) = get_1d_rotary_pos_embed(h_dim, max_seq_len, theta, device)?;
        let (cos_w, sin_w) = get_1d_rotary_pos_embed(w_dim, max_seq_len, theta, device)?;

        // Concatenate along dim=1
        let freqs_cos = Tensor::cat(&[&cos_t, &cos_h, &cos_w], 1)?;
        let freqs_sin = Tensor::cat(&[&sin_t, &sin_h, &sin_w], 1)?;

        Ok(Self {
            attention_head_dim,
            patch_size,
            max_seq_len,
            theta,
            freqs_cos,
            freqs_sin,
            t_dim,
            h_dim,
            w_dim,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_b, _c, frames, height, width) = hidden_states.dims5()?;
        let (p_t, p_h, p_w) = self.patch_size;

        let ppf = frames / p_t;
        let pph = height / p_h;
        let ppw = width / p_w;

        // Split frequencies by axis dimensions using narrow
        // freqs_cos shape: [max_seq_len, t_dim + h_dim + w_dim]
        let cos_t = self.freqs_cos.narrow(0, 0, ppf)?.narrow(1, 0, self.t_dim)?;
        let cos_h = self
            .freqs_cos
            .narrow(0, 0, pph)?
            .narrow(1, self.t_dim, self.h_dim)?;
        let cos_w =
            self.freqs_cos
                .narrow(0, 0, ppw)?
                .narrow(1, self.t_dim + self.h_dim, self.w_dim)?;

        let sin_t = self.freqs_sin.narrow(0, 0, ppf)?.narrow(1, 0, self.t_dim)?;
        let sin_h = self
            .freqs_sin
            .narrow(0, 0, pph)?
            .narrow(1, self.t_dim, self.h_dim)?;
        let sin_w =
            self.freqs_sin
                .narrow(0, 0, ppw)?
                .narrow(1, self.t_dim + self.h_dim, self.w_dim)?;

        // Reshape and broadcast for 3D grid
        let cos_t = cos_t
            .reshape((ppf, 1, 1, self.t_dim))?
            .broadcast_as((ppf, pph, ppw, self.t_dim))?;
        let cos_h = cos_h
            .reshape((1, pph, 1, self.h_dim))?
            .broadcast_as((ppf, pph, ppw, self.h_dim))?;
        let cos_w = cos_w
            .reshape((1, 1, ppw, self.w_dim))?
            .broadcast_as((ppf, pph, ppw, self.w_dim))?;

        let sin_t = sin_t
            .reshape((ppf, 1, 1, self.t_dim))?
            .broadcast_as((ppf, pph, ppw, self.t_dim))?;
        let sin_h = sin_h
            .reshape((1, pph, 1, self.h_dim))?
            .broadcast_as((ppf, pph, ppw, self.h_dim))?;
        let sin_w = sin_w
            .reshape((1, 1, ppw, self.w_dim))?
            .broadcast_as((ppf, pph, ppw, self.w_dim))?;

        // Concatenate along last dim
        let cos = Tensor::cat(&[&cos_t, &cos_h, &cos_w], 3)?;
        let sin = Tensor::cat(&[&sin_t, &sin_h, &sin_w], 3)?;

        // Reshape to [1, seq, 1, head_dim]
        let seq = ppf * pph * ppw;
        let cos = cos.reshape((1, seq, 1, self.attention_head_dim))?;
        let sin = sin.reshape((1, seq, 1, self.attention_head_dim))?;

        Ok((cos, sin))
    }
}

// ============================================================================
// WanTimeTextImageEmbedding
// ============================================================================

/// Condition embedder for timestep, text, and optional image embeddings.
#[derive(Clone, Debug)]
pub struct WanTimeTextImageEmbedding {
    #[allow(dead_code)]
    dim: usize,
    time_embedder: TimestepEmbedding,
    time_proj: Linear,
    text_embedder: PixArtAlphaTextProjection,
    freq_dim: usize,
    // I2V image embedder (commented out for T2V-only)
    // image_embedder: Option<WanImageEmbedding>,
}

impl WanTimeTextImageEmbedding {
    pub fn new(
        dim: usize,
        time_freq_dim: usize,
        time_proj_dim: usize,
        text_embed_dim: usize,
        _image_embed_dim: Option<usize>,
        _pos_embed_seq_len: Option<usize>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let time_embedder = TimestepEmbedding::new(time_freq_dim, dim, vb.pp("time_embedder"))?;
        let time_proj = nn::linear(dim, time_proj_dim, vb.pp("time_proj"))?;
        let text_embedder =
            PixArtAlphaTextProjection::new(text_embed_dim, dim, vb.pp("text_embedder"))?;

        // I2V support commented out
        // let image_embedder = if let Some(img_dim) = image_embed_dim {
        //     Some(WanImageEmbedding::new(img_dim, dim, pos_embed_seq_len, vb.pp("image_embedder"))?)
        // } else {
        //     None
        // };

        Ok(Self {
            dim,
            time_embedder,
            time_proj,
            text_embedder,
            freq_dim: time_freq_dim,
            // image_embedder,
        })
    }

    pub fn forward(
        &self,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        _encoder_hidden_states_image: Option<&Tensor>,
        timestep_seq_len: Option<usize>,
    ) -> CandleResult<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        // Timestep projection: sinusoidal -> time_embedder -> SiLU -> time_proj
        let timestep_proj = get_timestep_embedding(timestep, self.freq_dim, true)?;

        // Handle Wan 2.2 ti2v case where timestep has seq_len dimension
        let timestep_proj = if let Some(seq_len) = timestep_seq_len {
            let batch = timestep_proj.dim(0)? / seq_len;
            timestep_proj.reshape((batch, seq_len, self.freq_dim))?
        } else {
            timestep_proj
        };

        let temb = self.time_embedder.forward(&timestep_proj)?;
        let temb = temb.to_dtype(encoder_hidden_states.dtype())?;
        let timestep_proj_out = self.time_proj.forward(&temb.silu()?)?;

        // Text embedding projection
        let enc_text = self.text_embedder.forward(encoder_hidden_states)?;

        // I2V image embedding (commented out)
        let enc_img: Option<Tensor> = None;
        // if let (Some(embedder), Some(img)) = (&self.image_embedder, encoder_hidden_states_image) {
        //     enc_img = Some(embedder.forward(img)?);
        // }

        Ok((temb, timestep_proj_out, enc_text, enc_img))
    }
}

// ============================================================================
// WanTransformerBlock
// ============================================================================

/// Single transformer block with self-attention, cross-attention, and FFN.
#[derive(Clone, Debug)]
pub struct WanTransformerBlock {
    norm1: FP32LayerNorm,
    attn1: WanAttention,
    attn2: WanAttention,
    norm2: Option<FP32LayerNorm>,
    ffn: WanFeedForward,
    norm3: FP32LayerNorm,
    scale_shift_table: Tensor,
}

impl WanTransformerBlock {
    pub fn new(
        dim: usize,
        ffn_dim: usize,
        num_heads: usize,
        cross_attn_norm: bool,
        eps: f64,
        _added_kv_proj_dim: Option<usize>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let dim_head = dim / num_heads;

        // Self-attention (diffusers: attn1)
        let norm1 = FP32LayerNorm::new(dim, eps, false, vb.pp("norm1"))?;
        let attn1 = WanAttention::new(dim, num_heads, dim_head, eps, None, vb.pp("attn1"))?;

        // Cross-attention (diffusers: attn2)
        let attn2 = WanAttention::new(
            dim,
            num_heads,
            dim_head,
            eps,
            Some(dim_head),
            vb.pp("attn2"),
        )?;
        
        // Note: diffusers swaps norm2/norm3 compared to official
        // diffusers norm2 = cross_attn_norm, norm3 = ffn norm
        let norm2 = if cross_attn_norm {
            Some(FP32LayerNorm::new(dim, eps, true, vb.pp("norm2"))?)
        } else {
            None
        };

        // Feed-forward
        let ffn = WanFeedForward::new(dim, ffn_dim, vb.pp("ffn"))?;
        let norm3 = FP32LayerNorm::new(dim, eps, false, vb.pp("norm3"))?;

        // Scale-shift table (diffusers name for modulation)
        let scale_shift_table = vb.get((1, 6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            attn1,
            attn2,
            norm2,
            ffn,
            norm3,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb_proj: &Tensor,
        rotary_emb: (&Tensor, &Tensor),
    ) -> CandleResult<Tensor> {
        let dtype = hidden_states.dtype();

        // Compute scale/shift/gate from temb_proj
        let (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa) =
            self.compute_scale_shift_gates(temb_proj)?;

        // 1. Self-attention
        let norm_h = self.norm1.forward(&hidden_states.to_dtype(DType::F32)?)?;
        let ones = Tensor::ones_like(&scale_msa)?;
        let scale_plus_one = scale_msa.broadcast_add(&ones)?;
        let norm_h = norm_h
            .broadcast_mul(&scale_plus_one)?
            .broadcast_add(&shift_msa)?;
        let norm_h = norm_h.to_dtype(dtype)?;
        let attn_out = self.attn1.forward(&norm_h, None, None, Some(rotary_emb))?;
        let hs = hidden_states
            .to_dtype(DType::F32)?
            .broadcast_add(&attn_out.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?
            .to_dtype(dtype)?;

        // 2. Cross-attention
        let norm_h2 = if let Some(n2) = &self.norm2 {
            n2.forward(&hs.to_dtype(DType::F32)?)?.to_dtype(dtype)?
        } else {
            hs.clone()
        };
        let attn_out2 = self
            .attn2
            .forward(&norm_h2, Some(encoder_hidden_states), None, None)?;
        let hs = hs.broadcast_add(&attn_out2)?;

        // 3. Feed-forward
        let norm_h3 = self.norm3.forward(&hs.to_dtype(DType::F32)?)?;
        let c_ones = Tensor::ones_like(&c_scale_msa)?;
        let c_scale_plus_one = c_scale_msa.broadcast_add(&c_ones)?;
        let norm_h3 = norm_h3
            .broadcast_mul(&c_scale_plus_one)?
            .broadcast_add(&c_shift_msa)?;
        let norm_h3 = norm_h3.to_dtype(dtype)?;
        let ff = self.ffn.forward(&norm_h3)?;
        let hs = hs
            .to_dtype(DType::F32)?
            .broadcast_add(&ff.to_dtype(DType::F32)?.broadcast_mul(&c_gate_msa)?)?
            .to_dtype(dtype)?;

        Ok(hs)
    }

    fn compute_scale_shift_gates(
        &self,
        temb: &Tensor,
    ) -> CandleResult<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let temb_f = temb.to_dtype(DType::F32)?;
        let sst = self.scale_shift_table.to_dtype(DType::F32)?;

        match temb_f.rank() {
            3 => {
                // [b, 6, dim] - standard Wan 2.1 case
                let sst = sst.broadcast_as(temb_f.dims())?;
                let x = sst.broadcast_add(&temb_f)?;
                let parts = x.chunk(6, 1)?;
                let mut p = Vec::with_capacity(6);
                for t in parts {
                    p.push(t.squeeze(1)?.unsqueeze(1)?); // [b, 1, dim]
                }
                Ok((
                    p[0].clone(),
                    p[1].clone(),
                    p[2].clone(),
                    p[3].clone(),
                    p[4].clone(),
                    p[5].clone(),
                ))
            }
            4 => {
                // [b, s, 6, dim] - Wan 2.2 ti2v case (commented out for now)
                let (b, s, _, dim) = temb_f.dims4()?;
                let sst = sst.unsqueeze(0)?.broadcast_as((b, s, 6, dim))?;
                let x = sst.broadcast_add(&temb_f)?;
                let parts = x.chunk(6, 2)?;
                let mut p = Vec::with_capacity(6);
                for t in parts {
                    p.push(t.squeeze(2)?); // [b, s, dim]
                }
                Ok((
                    p[0].clone(),
                    p[1].clone(),
                    p[2].clone(),
                    p[3].clone(),
                    p[4].clone(),
                    p[5].clone(),
                ))
            }
            _ => Err(candle_core::Error::Msg(
                "temb_proj must be rank 3 ([b,6,dim]) or rank 4 ([b,s,6,dim])".to_string(),
            )),
        }
    }
}

// ============================================================================
// WanTransformer3DModel
// ============================================================================

/// Output type for transformer forward pass.
#[derive(Debug)]
pub struct Transformer2DModelOutput {
    pub sample: Tensor,
}

/// Main Wan Transformer 3D model.
pub struct WanTransformer3DModel {
    pub cfg: WanTransformer3DConfig,
    rope: WanRotaryPosEmbed,
    patch_embedding: Conv3d,
    condition_embedder: WanTimeTextImageEmbedding,
    blocks: Vec<WanTransformerBlock>,
    norm_out: FP32LayerNorm,
    proj_out: Linear,
    scale_shift_table: Tensor,
}

impl WanTransformer3DModel {
    /// Create a new WanTransformer3DModel with VarBuilder for weight loading.
    pub fn new(cfg: WanTransformer3DConfig, vb: VarBuilder) -> CandleResult<Self> {
        let inner_dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let device = vb.device();

        // 1. Rotary position embeddings
        let rope = WanRotaryPosEmbed::new(
            cfg.attention_head_dim,
            cfg.patch_size,
            cfg.rope_max_seq_len,
            10000.0,
            device,
        )?;

        // 2. Patch embedding (Conv3d with kernel=stride=patch_size)
        let patch_cfg = Conv3dConfig {
            stride: cfg.patch_size,
            padding: (0, 0, 0),
            ..Default::default()
        };
        let patch_embedding = Conv3d::new(
            cfg.in_channels,
            inner_dim,
            cfg.patch_size,
            patch_cfg,
            vb.pp("patch_embedding"),
        )?;

        // 3. Condition embedder
        let condition_embedder = WanTimeTextImageEmbedding::new(
            inner_dim,
            cfg.freq_dim,
            inner_dim * 6,
            cfg.text_dim,
            cfg.image_dim,
            None, // pos_embed_seq_len
            vb.pp("condition_embedder"),
        )?;

        // 4. Transformer blocks
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = WanTransformerBlock::new(
                inner_dim,
                cfg.ffn_dim,
                cfg.num_attention_heads,
                cfg.cross_attn_norm,
                cfg.eps,
                cfg.added_kv_proj_dim,
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        // 5. Output norm and projection
        let norm_out = FP32LayerNorm::new(inner_dim, cfg.eps, false, vb.pp("norm_out"))?;
        let out_channels = cfg.out_channels;
        let patch_prod = cfg.patch_size.0 * cfg.patch_size.1 * cfg.patch_size.2;
        let proj_out = nn::linear(inner_dim, out_channels * patch_prod, vb.pp("proj_out"))?;

        // 6. Scale-shift table for output: [1, 2, inner_dim]
        let scale_shift_table = vb.get((1, 2, inner_dim), "scale_shift_table")?;

        Ok(Self {
            cfg,
            rope,
            patch_embedding,
            condition_embedder,
            blocks,
            norm_out,
            proj_out,
            scale_shift_table,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.cfg.in_channels
    }

    pub fn dtype(&self) -> DType {
        self.proj_out.weight().dtype()
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_hidden_states_image: Option<&Tensor>,
        return_dict: bool,
    ) -> CandleResult<std::result::Result<Transformer2DModelOutput, Tensor>> {
        let (b, _c, frames, height, width) = hidden_states.dims5()?;
        let (p_t, p_h, p_w) = self.cfg.patch_size;
        let inner_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim;

        let post_f = frames / p_t;
        let post_h = height / p_h;
        let post_w = width / p_w;

        // 1. Rotary embeddings
        let (freqs_cos, freqs_sin) = self.rope.forward(hidden_states)?;

        // 2. Patch embedding
        let hs = self.patch_embedding.forward(hidden_states)?;
        let hs = hs.flatten_from(2)?; // [b, inner_dim, seq]
        let mut hs = hs.transpose(1, 2)?; // [b, seq, inner_dim]

        // 3. Timestep handling
        // Wan 2.2 ti2v: timestep.ndim==2 => flatten and track seq_len
        let (timestep_flat, ts_seq_len) = match timestep.rank() {
            1 => (timestep.clone(), None),
            2 => {
                let (_, s2) = timestep.dims2()?;
                (timestep.flatten_all()?, Some(s2))
            }
            _ => {
                return Err(candle_core::Error::Msg(
                    "timestep must be rank-1 [b] or rank-2 [b, seq_len]".to_string(),
                ));
            }
        };

        // 4. Condition embedder
        let (temb, timestep_proj, enc_text, enc_img) = self.condition_embedder.forward(
            &timestep_flat,
            encoder_hidden_states,
            encoder_hidden_states_image,
            ts_seq_len,
        )?;

        // Reshape timestep_proj to [b, 6, inner_dim] or [b, seq, 6, inner_dim]
        let timestep_proj = match timestep_proj.rank() {
            2 => {
                let (bb, _) = timestep_proj.dims2()?;
                timestep_proj.reshape((bb, 6, inner_dim))?
            }
            3 => {
                let (bb, ss, _) = timestep_proj.dims3()?;
                timestep_proj.reshape((bb, ss, 6, inner_dim))?
            }
            _ => {
                return Err(candle_core::Error::Msg(
                    "timestep_proj must be rank-2 or rank-3".to_string(),
                ));
            }
        };

        // Concatenate image and text embeddings if present
        let enc = if let Some(img) = enc_img {
            Tensor::cat(&[&img, &enc_text], 1)?
        } else {
            enc_text
        };

        // 5. Transformer blocks
        for blk in &self.blocks {
            hs = blk.forward(&hs, &enc, &timestep_proj, (&freqs_cos, &freqs_sin))?;
        }

        // 6. Output norm and projection
        let (shift, scale) = self.compute_out_shift_scale(&temb, inner_dim)?;

        let hs_norm = self.norm_out.forward(&hs.to_dtype(DType::F32)?)?;
        let ones_scale = Tensor::ones_like(&scale)?;
        let hs_norm = hs_norm
            .broadcast_mul(&scale.broadcast_add(&ones_scale)?)?
            .broadcast_add(&shift)?;
        let hs_norm = hs_norm.to_dtype(hs.dtype())?;

        let out = self.proj_out.forward(&hs_norm)?;

        // 7. Unpatchify - split into two reshape/permute operations since candle doesn't support 8D
        let out_ch = self.cfg.out_channels;
        // First reshape to 6D: [b, post_f*post_h, post_w, p_t*p_h, p_w, out_ch]
        let out = out.reshape(&[b, post_f * post_h, post_w, p_t * p_h, p_w, out_ch])?;
        // Permute to [b, out_ch, post_f*post_h, p_t*p_h, post_w, p_w]
        let out = out.permute([0, 5, 1, 3, 2, 4])?;
        // Flatten to get final shape [b, out_ch, post_f*p_t*post_h*p_h, post_w*p_w]
        let out = out.flatten(4, 5)?; // merge post_w, p_w
        let out = out.flatten(2, 3)?; // merge post_f*post_h, p_t*p_h
        // Reshape to [b, out_ch, frames, height, width]
        let out = out.reshape(&[b, out_ch, post_f * p_t, post_h * p_h, post_w * p_w])?;

        if return_dict {
            Ok(Ok(Transformer2DModelOutput { sample: out }))
        } else {
            Ok(Err(out))
        }
    }

    fn compute_out_shift_scale(
        &self,
        temb: &Tensor,
        inner_dim: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let sst = self.scale_shift_table.to_dtype(DType::F32)?;

        match temb.rank() {
            2 => {
                // temb: [b, inner_dim]
                let (b, _) = temb.dims2()?;
                let t = temb.to_dtype(DType::F32)?.unsqueeze(1)?; // [b, 1, inner]
                let sst = sst.broadcast_as((b, 2, inner_dim))?;
                let x = sst.broadcast_add(&t)?;
                let parts = x.chunk(2, 1)?;
                let shift = parts[0].squeeze(1)?.unsqueeze(1)?; // [b, 1, inner]
                let scale = parts[1].squeeze(1)?.unsqueeze(1)?;
                Ok((shift, scale))
            }
            3 => {
                // temb: [b, seq, inner_dim] - Wan 2.2 ti2v
                let (b, s, _) = temb.dims3()?;
                let t = temb.to_dtype(DType::F32)?.unsqueeze(2)?; // [b, s, 1, inner]
                let sst = sst.unsqueeze(0)?.broadcast_as((b, s, 2, inner_dim))?;
                let x = sst.broadcast_add(&t)?;
                let parts = x.chunk(2, 2)?;
                let shift = parts[0].squeeze(2)?; // [b, s, inner]
                let scale = parts[1].squeeze(2)?;
                Ok((shift, scale))
            }
            _ => Err(candle_core::Error::Msg(
                "temb must be rank-2 ([b,inner]) or rank-3 ([b,seq,inner])".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp32_layer_norm() -> CandleResult<()> {
        let device = Device::Cpu;
        let xs = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
        let norm = FP32LayerNorm::new_no_affine(1e-5);
        let out = norm.forward(&xs)?;
        assert_eq!(out.dims(), xs.dims());
        Ok(())
    }

    #[test]
    fn test_wan_rotary_pos_embed() -> CandleResult<()> {
        let device = Device::Cpu;
        let rope = WanRotaryPosEmbed::new(128, (1, 2, 2), 1024, 10000.0, &device)?;
        let hidden = Tensor::zeros((1, 16, 4, 32, 32), DType::F32, &device)?;
        let (cos, sin) = rope.forward(&hidden)?;
        // seq = (4/1) * (32/2) * (32/2) = 4 * 16 * 16 = 1024
        assert_eq!(cos.dims(), &[1, 1024, 1, 128]);
        assert_eq!(sin.dims(), &[1, 1024, 1, 128]);
        Ok(())
    }
}
