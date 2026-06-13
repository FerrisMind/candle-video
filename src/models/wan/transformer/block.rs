//! Wan transformer block (`WanTransformerBlock`).

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::models::wan::configs::WanTransformerConfig;

use super::attention::WanAttention;
use super::feed_forward::WanFeedForward;
use super::fp32_layer_norm::Fp32LayerNorm;

#[derive(Debug)]
pub struct WanTransformerBlock {
    norm1: Fp32LayerNorm,
    attn1: WanAttention,
    norm2: Fp32LayerNorm,
    attn2: WanAttention,
    norm3: Fp32LayerNorm,
    ffn: WanFeedForward,
    scale_shift_table: Tensor,
    cross_attn_norm: bool,
}

impl WanTransformerBlock {
    pub fn new(cfg: &WanTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.num_attention_heads * cfg.attention_head_dim;
        let head_dim = cfg.attention_head_dim;
        let cross_norm = if cfg.cross_attn_norm {
            Fp32LayerNorm::new(dim, cfg.eps, true, vb.pp("norm2"))?
        } else {
            Fp32LayerNorm::new(dim, cfg.eps, false, vb.pp("norm2"))?
        };

        Ok(Self {
            norm1: Fp32LayerNorm::new(dim, cfg.eps, false, vb.pp("norm1"))?,
            attn1: WanAttention::new_self_attn(
                dim,
                cfg.num_attention_heads,
                head_dim,
                cfg.eps,
                vb.pp("attn1"),
            )?,
            norm2: cross_norm,
            attn2: WanAttention::new_cross_attn(
                dim,
                cfg.num_attention_heads,
                head_dim,
                cfg.eps,
                vb.pp("attn2"),
            )?,
            norm3: Fp32LayerNorm::new(dim, cfg.eps, false, vb.pp("norm3"))?,
            ffn: WanFeedForward::new(dim, cfg.ffn_dim, vb.pp("ffn"))?,
            scale_shift_table: vb.get((1, 6, dim), "scale_shift_table")?,
            cross_attn_norm: cfg.cross_attn_norm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep_proj: &Tensor,
        rotary_emb: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let dim = hidden_states.dim(2)?;
        let table = self.scale_shift_table.to_dtype(DType::F32)?;
        let temb = timestep_proj.to_dtype(DType::F32)?;
        let combined = table.broadcast_add(&temb)?;
        let shift_msa = combined.narrow(1, 0, 1)?.squeeze(1)?;
        let scale_msa = combined.narrow(1, 1, 1)?.squeeze(1)?;
        let gate_msa = combined.narrow(1, 2, 1)?.squeeze(1)?;
        let c_shift_msa = combined.narrow(1, 3, 1)?.squeeze(1)?;
        let c_scale_msa = combined.narrow(1, 4, 1)?.squeeze(1)?;
        let c_gate_msa = combined.narrow(1, 5, 1)?.squeeze(1)?;

        let hs_f32 = hidden_states.to_dtype(DType::F32)?;
        let norm_hidden = self.norm1.forward(&hs_f32)?;
        let ones = Tensor::ones_like(&norm_hidden)?;
        let norm_hidden = norm_hidden.broadcast_mul(&(scale_msa.broadcast_add(&ones)?))?;
        let norm_hidden = norm_hidden.broadcast_add(&shift_msa)?;
        let norm_hidden = norm_hidden.to_dtype(hidden_states.dtype())?;

        let attn_out = self
            .attn1
            .forward(&norm_hidden, None, Some(rotary_emb))?;
        let mut hidden_states = (hs_f32 + attn_out.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?
            .to_dtype(hidden_states.dtype())?;

        let norm_hidden = if self.cross_attn_norm {
            self.norm2.forward(&hidden_states.to_dtype(DType::F32)?)?
                .to_dtype(hidden_states.dtype())?
        } else {
            hidden_states.clone()
        };
        let attn_out = self.attn2.forward(&norm_hidden, Some(encoder_hidden_states), None)?;
        hidden_states = (hidden_states.to_dtype(DType::F32)? + attn_out.to_dtype(DType::F32)?)?
            .to_dtype(hidden_states.dtype())?;

        let hs_f32 = hidden_states.to_dtype(DType::F32)?;
        let norm_hidden = self.norm3.forward(&hs_f32)?;
        let ones = Tensor::ones_like(&norm_hidden)?;
        let norm_hidden = norm_hidden.broadcast_mul(&(c_scale_msa.broadcast_add(&ones)?))?;
        let norm_hidden = norm_hidden.broadcast_add(&c_shift_msa)?;
        let ff_out = self.ffn.forward(&norm_hidden.to_dtype(hidden_states.dtype())?)?;
        hidden_states = (hs_f32 + ff_out.to_dtype(DType::F32)?.broadcast_mul(&c_gate_msa)?)?
            .to_dtype(hidden_states.dtype())?;

        let _ = dim;
        Ok(hidden_states)
    }
}
