//! Spatio-Temporal Transformer for SVD UNet
//!
//! Implements spatial and temporal attention mechanisms.

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear};

use super::model::get_timestep_embedding;

/// Feed-forward network with GEGLU activation
#[derive(Debug)]
struct FeedForward {
    proj: Linear,
    proj_out: Linear,
    inner_dim: usize,
}

impl FeedForward {
    fn new(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let proj = linear(dim, inner_dim * 2, vb.pp("net").pp("0").pp("proj"))?;
        let proj_out = linear(inner_dim, dim, vb.pp("net").pp("2"))?;
        Ok(Self {
            proj,
            proj_out,
            inner_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(x)?;
        let gate = h.narrow(D::Minus1, 0, self.inner_dim)?;
        let value = h.narrow(D::Minus1, self.inner_dim, self.inner_dim)?;
        let h = (gate.gelu_erf()? * value)?;
        self.proj_out.forward(&h)
    }
}

/// Cross/Self Attention
#[derive(Debug)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        query_dim: usize,
        heads: usize,
        dim_head: usize,
        kv_dim: Option<usize>,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let kv_in_dim = kv_dim.unwrap_or(query_dim);

        let to_q = candle_nn::linear_no_bias(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_no_bias(kv_in_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_no_bias(kv_in_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out").pp("0"))?;

        let scale = (dim_head as f64).powf(-0.5);

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            head_dim: dim_head,
            scale,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let context = encoder_hidden_states.unwrap_or(hidden_states);
        let (batch, seq_len, _) = hidden_states.dims3()?;

        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;

        let q = q
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, (), self.heads, self.head_dim))?
            .transpose(1, 2)?
            .transpose(2, 3)? // [B, H, D, S_k] for matmul
            .contiguous()?;
        let v = v
            .reshape((batch, (), self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Sliced attention calculation to avoid OOM
        // Split q into chunks along sequence dimension
        // Chunk 32 approx 330MB peak overhead for attention scores in F32 (safest) (Still OOM on some 3060s)
        // Chunk 2 approx 20MB -> Should definitely fit even with fragmentation
        let chunk_size = 2;
        let mut chunks = Vec::new();

        for i in (0..seq_len).step_by(chunk_size) {
            let end = std::cmp::min(i + chunk_size, seq_len);
            let q_chunk = q.narrow(2, i, end - i)?; // [B, H, Chunk, D]

            // Compute attention in F32 for numerical stability in FP16 mode
            // Cast Q and K to F32 before matmul to prevent overflow
            let q_f32 = q_chunk.to_dtype(DType::F32)?;
            let k_f32 = k.to_dtype(DType::F32)?;
            let attn_weights = (q_f32.matmul(&k_f32)? * self.scale)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            
            // Cast V to F32 for matmul, then back to original dtype
            let v_f32 = v.to_dtype(DType::F32)?;
            let out_chunk = attn_weights.matmul(&v_f32)?.to_dtype(q.dtype())?;
            chunks.push(out_chunk);
        }

        let out = Tensor::cat(&chunks, 2)?; // [B, H, S, D]

        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, ()))?
            .to_dtype(hidden_states.dtype())?;
        self.to_out.forward(&out)
    }
}

/// Basic transformer block
#[derive(Debug)]
pub struct BasicTransformerBlock {
    norm1: candle_nn::LayerNorm,
    attn1: Attention,
    norm2: candle_nn::LayerNorm,
    attn2: Attention,
    norm3: candle_nn::LayerNorm,
    ff: FeedForward,
}

impl BasicTransformerBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        cross_attention_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?,
            attn1: Attention::new(vb.pp("attn1"), dim, num_heads, head_dim, None)?,
            norm2: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?,
            attn2: Attention::new(
                vb.pp("attn2"),
                dim,
                num_heads,
                head_dim,
                Some(cross_attention_dim),
            )?,
            norm3: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm3"))?,
            ff: FeedForward::new(vb.pp("ff"), dim, 4)?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let h = self
            .attn1
            .forward(&self.norm1.forward(hidden_states)?, None)?;
        let h = (h + residual)?;

        let residual = &h;
        let h = self
            .attn2
            .forward(&self.norm2.forward(&h)?, encoder_hidden_states)?;
        let h = (h + residual)?;

        let residual = &h;
        let h = self.ff.forward(&self.norm3.forward(&h)?)?;
        h + residual
    }
}

/// Temporal transformer block
#[derive(Debug)]
pub struct TemporalBasicTransformerBlock {
    norm_in: candle_nn::LayerNorm,
    ff_in: FeedForward,
    norm1: candle_nn::LayerNorm,
    attn1: Attention,
    norm2: candle_nn::LayerNorm,
    attn2: Attention,
    norm3: candle_nn::LayerNorm,
    ff: FeedForward,
}

impl TemporalBasicTransformerBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        cross_attention_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm_in: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm_in"))?,
            ff_in: FeedForward::new(vb.pp("ff_in"), dim, 4)?,
            norm1: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?,
            attn1: Attention::new(vb.pp("attn1"), dim, num_heads, head_dim, None)?,
            norm2: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?,
            attn2: Attention::new(
                vb.pp("attn2"),
                dim,
                num_heads,
                head_dim,
                Some(cross_attention_dim),
            )?,
            norm3: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm3"))?,
            ff: FeedForward::new(vb.pp("ff"), dim, 4)?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,  // [B*S, seq_len, D] where S=H*W
        num_frames: usize,
    ) -> Result<Tensor> {
        let (batch_frames, seq_len, dim) = hidden_states.dims3()?;
        let batch_size = batch_frames / num_frames;

        // Reshape for temporal: [B*F, S, C] -> [B*S, F, C]
        // where F = num_frames, S = seq_len (H*W)
        let h = hidden_states
            .reshape((batch_size, num_frames, seq_len, dim))?
            .permute((0, 2, 1, 3))?  // [B, S, F, C]
            .reshape((batch_size * seq_len, num_frames, dim))?;  // [B*S, F, C]

        // encoder_hidden_states is already [B*S, enc_seq_len, D] from parent transformer
        // It will be used for cross-attention where Q has seq_len=num_frames, K/V have seq_len=enc_seq_len

        // FF_in with residual connection
        let residual = &h;
        let h = self.ff_in.forward(&self.norm_in.forward(&h)?)?;
        // is_res = True in diffusers (dim == time_mix_inner_dim for SVD)
        let h = (h + residual)?;

        // Self-attention
        let residual = &h;
        let h = self.attn1.forward(&self.norm1.forward(&h)?, None)?;
        let h = (h + residual)?;

        // Cross-attention (with encoder_hidden_states as context)
        let residual = &h;
        let h = self
            .attn2
            .forward(&self.norm2.forward(&h)?, encoder_hidden_states)?;
        let h = (h + residual)?;

        // FF with residual
        let residual = &h;
        let h = self.ff.forward(&self.norm3.forward(&h)?)?;
        // is_res = True in diffusers
        let h = (h + residual)?;

        // Reshape back: [B*S, F, C] -> [B*F, S, C]
        h.reshape((batch_size, seq_len, num_frames, dim))?
            .permute((0, 2, 1, 3))?  // [B, F, S, C]
            .reshape((batch_frames, seq_len, dim))
    }
}

/// Time position embedding
#[derive(Debug)]
struct TimePosEmbed {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimePosEmbed {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let inner_dim = dim * 4;
        Ok(Self {
            linear_1: linear(dim, inner_dim, vb.pp("linear_1"))?,
            linear_2: linear(inner_dim, dim, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_nn::ops::silu(&self.linear_1.forward(x)?)?;
        self.linear_2.forward(&h)
    }
}

/// Time mixer
#[derive(Debug)]
struct TimeMixer {
    mix_factor: Tensor,
}

impl TimeMixer {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            mix_factor: vb.get(1, "mix_factor")?,
        })
    }

    fn forward(&self, x_spatial: &Tensor, x_temporal: &Tensor) -> Result<Tensor> {
        let alpha = candle_nn::ops::sigmoid(&self.mix_factor)?.broadcast_as(x_spatial.shape())?;
        (x_temporal * &alpha)? + (x_spatial * (1.0 - &alpha)?)?
    }
}

/// Full Spatio-Temporal Transformer
#[derive(Debug)]
pub struct TransformerSpatioTemporalModel {
    norm: candle_nn::GroupNorm,
    proj_in: Linear,
    transformer_blocks: Vec<BasicTransformerBlock>,
    temporal_transformer_blocks: Vec<TemporalBasicTransformerBlock>,
    time_pos_embed: TimePosEmbed,
    time_mixer: TimeMixer,
    proj_out: Linear,
    in_channels: usize,
}

impl TransformerSpatioTemporalModel {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        num_layers: usize,
        num_heads: usize,
        cross_attention_dim: usize,
    ) -> Result<Self> {
        let head_dim = in_channels / num_heads;

        let norm = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm"))?;
        let proj_in = linear(in_channels, in_channels, vb.pp("proj_in"))?;

        let mut transformer_blocks = Vec::with_capacity(num_layers);
        let mut temporal_transformer_blocks = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            transformer_blocks.push(BasicTransformerBlock::new(
                vb.pp("transformer_blocks").pp(i),
                in_channels,
                num_heads,
                head_dim,
                cross_attention_dim,
            )?);
            temporal_transformer_blocks.push(TemporalBasicTransformerBlock::new(
                vb.pp("temporal_transformer_blocks").pp(i),
                in_channels,
                num_heads,
                head_dim,
                cross_attention_dim,
            )?);
        }

        let time_pos_embed = TimePosEmbed::new(vb.pp("time_pos_embed"), in_channels)?;
        let time_mixer = TimeMixer::new(vb.pp("time_mixer"))?;
        let proj_out = linear(in_channels, in_channels, vb.pp("proj_out"))?;

        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            temporal_transformer_blocks,
            time_pos_embed,
            time_mixer,
            proj_out,
            in_channels,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<Tensor> {
        let (batch_frames, c, h, w) = hidden_states.dims4()?;
        let batch_size = batch_frames / num_frames;
        let residual = hidden_states;

        // Prepare time_context for temporal cross-attention from encoder_hidden_states
        // diffusers: takes first frame's encoder states, broadcasts it to all H*W positions
        let time_context = encoder_hidden_states.map(|ehs| {
            // ehs: [B*T, 1, D] -> reshape to [B, T, 1, D] -> take first frame [:, 0] -> [B, 1, D]
            let d = ehs.dim(2).unwrap();
            let first_frame_ehs = ehs
                .reshape((batch_size, num_frames, 1, d))
                .unwrap()
                .narrow(1, 0, 1)  // Take first frame
                .unwrap()
                .squeeze(1)  // [B, 1, D]
                .unwrap();
            // Broadcast to [B, H*W, 1, D] -> reshape to [B*H*W, 1, D]
            first_frame_ehs
                .unsqueeze(1)  // [B, 1, 1, D]
                .unwrap()
                .repeat((1, h * w, 1, 1))  // [B, H*W, 1, D]
                .unwrap()
                .reshape((batch_size * h * w, 1, d))  // [B*H*W, 1, D]
                .unwrap()
        });

        // Normalize and reshape to sequence
        let hidden_states = self
            .norm
            .forward(hidden_states)?
            .reshape((batch_frames, c, h * w))?
            .transpose(1, 2)?;

        let mut hidden_states = self.proj_in.forward(&hidden_states)?;

        // Create time position embedding for each frame
        // diffusers: time_proj(arange(num_frames)) -> [T, C] in F32, then cast to hidden_states dtype
        let num_frames_emb = Tensor::arange(0f32, num_frames as f32, hidden_states.device())?
            .repeat(batch_size)?  // [B*T]
            .reshape((batch_size * num_frames,))?;
        
        // get_timestep_embedding always returns F32, cast to match hidden_states
        let t_emb = get_timestep_embedding(&num_frames_emb, self.in_channels)?
            .to_dtype(hidden_states.dtype())?;

        let emb = self.time_pos_embed.forward(&t_emb)?;  // [B*T, C]
        let emb = emb.unsqueeze(1)?;  // [B*T, 1, C] for broadcast

        // Apply transformer blocks with mixing
        // diffusers order: spatial -> add emb -> temporal -> time_mixer
        for (spatial_block, temporal_block) in self
            .transformer_blocks
            .iter()
            .zip(&self.temporal_transformer_blocks)
        {
            // 1. Spatial block
            let h_spatial = spatial_block.forward(&hidden_states, encoder_hidden_states)?;
            
            // 2. Add time position embedding AFTER spatial block
            let hidden_states_mix = h_spatial.broadcast_add(&emb)?;
            
            // 3. Temporal block (uses time_context for cross-attention)
            let h_temporal = temporal_block.forward(&hidden_states_mix, time_context.as_ref(), num_frames)?;
            
            // 4. Mix spatial and temporal
            hidden_states = self.time_mixer.forward(&h_spatial, &h_temporal)?;
        }

        hidden_states = self.proj_out.forward(&hidden_states)?;

        // Reshape back
        let hidden_states = hidden_states
            .transpose(1, 2)?
            .reshape((batch_frames, c, h, w))?;
        hidden_states + residual
    }
}
