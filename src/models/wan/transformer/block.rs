//! Wan transformer block (`WanTransformerBlock`).

use std::sync::RwLock;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::attention::WanAttention;
use super::feed_forward::WanFeedForward;
use super::fp32_layer_norm::Fp32LayerNorm;
use super::rope::WanRotaryEmb;
use crate::models::wan::configs::WanTransformerConfig;

#[derive(Debug)]
pub struct WanTransformerBlock {
    norm1: Fp32LayerNorm,
    attn1: WanAttention,
    norm2: Fp32LayerNorm,
    attn2: WanAttention,
    norm3: Fp32LayerNorm,
    ffn: WanFeedForward,
    /// Source `scale_shift_table` weight, kept in its loaded dtype (typically F32).
    table_source: Tensor,
    /// Cached `scale_shift_table.to_dtype(dtype)` keyed on the activation dtype.
    /// ponytail: avoids per-forward `to_dtype` (was costing `memmove` CPU + GPU
    /// copies) — for 30 blocks × 3-50 steps that's hundreds of avoidable copies.
    table_cache: RwLock<Vec<(DType, Tensor)>>,
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
            table_source: vb.get((1, 6, dim), "scale_shift_table")?,
            table_cache: RwLock::new(Vec::new()),
            cross_attn_norm: cfg.cross_attn_norm,
        })
    }

    fn scale_shift_table(&self, dtype: DType) -> Result<Tensor> {
        if self.table_source.dtype() == dtype {
            return Ok(self.table_source.clone());
        }
        {
            let guard = self.table_cache.read().expect("table_cache poisoned");
            for (dt, t) in guard.iter() {
                if *dt == dtype {
                    return Ok(t.clone());
                }
            }
        }
        let mut guard = self.table_cache.write().expect("table_cache poisoned");
        // Re-check in case another thread raced us.
        for (dt, t) in guard.iter() {
            if *dt == dtype {
                return Ok(t.clone());
            }
        }
        let t = self.table_source.to_dtype(dtype)?;
        guard.push((dtype, t.clone()));
        Ok(t)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep_proj: &Tensor,
        rotary_emb: &WanRotaryEmb,
    ) -> Result<Tensor> {
        let dim = hidden_states.dim(2)?;
        if hidden_states.device().is_cuda() {
            return self.forward_cuda(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                dim,
            );
        }
        self.forward_f32(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            dim,
        )
    }

    /// CPU parity path — keeps activations in F32 between FP32LayerNorm and gated residuals.
    fn forward_f32(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep_proj: &Tensor,
        rotary_emb: &WanRotaryEmb,
        dim: usize,
    ) -> Result<Tensor> {
        let table = self.scale_shift_table(DType::F32)?;
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

        let attn_out = self.attn1.forward(&norm_hidden, None, Some(rotary_emb))?;
        let mut hidden_states = (hs_f32
            + attn_out.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?
            .to_dtype(hidden_states.dtype())?;

        let attn_out = if self.cross_attn_norm {
            let norm_hidden = self
                .norm2
                .forward(&hidden_states.to_dtype(DType::F32)?)?
                .to_dtype(hidden_states.dtype())?;
            self.attn2
                .forward(&norm_hidden, Some(encoder_hidden_states), None)?
        } else {
            self.attn2
                .forward(&hidden_states, Some(encoder_hidden_states), None)?
        };
        hidden_states = (hidden_states.to_dtype(DType::F32)? + attn_out.to_dtype(DType::F32)?)?
            .to_dtype(hidden_states.dtype())?;

        let hs_f32 = hidden_states.to_dtype(DType::F32)?;
        let norm_hidden = self.norm3.forward(&hs_f32)?;
        let ones = Tensor::ones_like(&norm_hidden)?;
        let norm_hidden = norm_hidden.broadcast_mul(&(c_scale_msa.broadcast_add(&ones)?))?;
        let norm_hidden = norm_hidden.broadcast_add(&c_shift_msa)?;
        let ff_out = self
            .ffn
            .forward(&norm_hidden.to_dtype(hidden_states.dtype())?)?;
        hidden_states = (hs_f32 + ff_out.to_dtype(DType::F32)?.broadcast_mul(&c_gate_msa)?)?
            .to_dtype(hidden_states.dtype())?;

        let _ = dim;
        Ok(hidden_states)
    }

    /// CUDA inference path — residuals/scale/gate in model dtype; FP32LayerNorm still normates in F32 internally.
    fn forward_cuda(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep_proj: &Tensor,
        rotary_emb: &WanRotaryEmb,
        dim: usize,
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let table = self.scale_shift_table(dtype)?;
        let temb = timestep_proj.to_dtype(dtype)?;
        let combined = table.broadcast_add(&temb)?;
        let shift_msa = combined.narrow(1, 0, 1)?.squeeze(1)?;
        let scale_msa = combined.narrow(1, 1, 1)?.squeeze(1)?;
        let gate_msa = combined.narrow(1, 2, 1)?.squeeze(1)?;
        let c_shift_msa = combined.narrow(1, 3, 1)?.squeeze(1)?;
        let c_scale_msa = combined.narrow(1, 4, 1)?.squeeze(1)?;
        let c_gate_msa = combined.narrow(1, 5, 1)?.squeeze(1)?;

        // ponytail: replace `Tensor::ones_like + broadcast_add + broadcast_mul`
        // with fused `affine(scale_msa + 1, shift_msa)`. Cuts one broadcast
        // and one full-tensor fill per block per step.
        let s_msa_1 = scale_msa.affine(1.0, 1.0)?;
        let mut norm_hidden = self.norm1.forward(hidden_states)?;
        norm_hidden = norm_hidden.broadcast_mul(&s_msa_1)?.broadcast_add(&shift_msa)?;

        let attn_out = self.attn1.forward(&norm_hidden, None, Some(rotary_emb))?;
        let attn_out = attn_out.broadcast_mul(&gate_msa)?;
        let mut hidden_states = hidden_states.add(&attn_out)?;

        let attn_out = if self.cross_attn_norm {
            let norm_hidden = self.norm2.forward(&hidden_states)?;
            self.attn2
                .forward(&norm_hidden, Some(encoder_hidden_states), None)?
        } else {
            self.attn2
                .forward(&hidden_states, Some(encoder_hidden_states), None)?
        };
        hidden_states = hidden_states.add(&attn_out)?;

        let c_s_msa_1 = c_scale_msa.affine(1.0, 1.0)?;
        norm_hidden = self.norm3.forward(&hidden_states)?;
        norm_hidden = norm_hidden.broadcast_mul(&c_s_msa_1)?.broadcast_add(&c_shift_msa)?;
        let ff_out = self.ffn.forward(&norm_hidden)?;
        let ff_out = ff_out.broadcast_mul(&c_gate_msa)?;
        let hidden_states = hidden_states.add(&ff_out)?;

        let _ = dim;
        Ok(hidden_states)
    }
}
