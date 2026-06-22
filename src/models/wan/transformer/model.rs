//! Full `WanTransformer3DModel` (T2V).

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, linear_b};

use crate::models::ltx_video::loader::{LoaderError, WeightLoader};
use crate::models::wan::configs::WanTransformerConfig;
use crate::models::wan::loader::{WanLayout, discover_safetensors, load_transformer_var_builder};

use super::block::WanTransformerBlock;
use super::embeddings::WanConditionEmbedder;
use super::fp32_layer_norm::Fp32LayerNorm;
use super::patch_embedding::WanPatchEmbedding;
use super::rope::{WanRotaryEmb, WanRotaryPosEmbed};

#[derive(Debug)]
pub struct WanTransformer3DModel {
    config: WanTransformerConfig,
    rope: WanRotaryPosEmbed,
    patch_embedding: WanPatchEmbedding,
    condition_embedder: WanConditionEmbedder,
    blocks: Vec<WanTransformerBlock>,
    norm_out: Fp32LayerNorm,
    proj_out: candle_nn::Linear,
    scale_shift_table: Tensor,
    inner_dim: usize,
}

impl WanTransformer3DModel {
    pub fn new(
        config: WanTransformerConfig,
        rope: WanRotaryPosEmbed,
        patch_embedding: WanPatchEmbedding,
        condition_embedder: WanConditionEmbedder,
        blocks: Vec<WanTransformerBlock>,
        norm_out: Fp32LayerNorm,
        proj_out: candle_nn::Linear,
        scale_shift_table: Tensor,
    ) -> Self {
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        Self {
            config,
            rope,
            patch_embedding,
            condition_embedder,
            blocks,
            norm_out,
            proj_out,
            scale_shift_table,
            inner_dim,
        }
    }

    pub fn from_var_builder(
        config: WanTransformerConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let out_per_patch = config.out_channels * config.patch_size.iter().product::<usize>();

        let rope = WanRotaryPosEmbed::new(&config, device)?;
        let patch_embedding = WanPatchEmbedding::new(
            config.in_channels,
            inner_dim,
            config.patch_size,
            vb.pp("patch_embedding"),
        )?;
        let condition_embedder = WanConditionEmbedder::new(
            inner_dim,
            config.freq_dim,
            config.text_dim,
            vb.pp("condition_embedder"),
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(WanTransformerBlock::new(
                &config,
                vb.pp("blocks").pp(i.to_string()),
            )?);
        }

        let norm_out = Fp32LayerNorm::new(inner_dim, config.eps, false, vb.pp("norm_out"))?;
        let proj_out = linear_b(inner_dim, out_per_patch, true, vb.pp("proj_out"))?;
        let scale_shift_table = vb.get((1, 2, inner_dim), "scale_shift_table")?;

        Ok(Self::new(
            config,
            rope,
            patch_embedding,
            condition_embedder,
            blocks,
            norm_out,
            proj_out,
            scale_shift_table,
        ))
    }

    pub fn load(
        transformer_dir: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, LoaderError> {
        let dir = transformer_dir.as_ref();
        let config: WanTransformerConfig =
            crate::models::ltx_video::loader::load_model_config(&dir.join("config.json"))?;
        let shards = discover_safetensors(dir)?;
        let loader = WeightLoader::new(device.clone(), dtype);
        let vb = if shards.len() == 1 {
            loader
                .load_single(&shards[0])
                .map_err(LoaderError::Candle)?
        } else {
            loader
                .load_sharded(shards.as_slice())
                .map_err(LoaderError::Candle)?
        };
        Self::from_var_builder(config, vb, device).map_err(LoaderError::Candle)
    }

    pub fn load_with_layout(
        layout: &WanLayout,
        device: &Device,
        dtype: DType,
        config: &WanTransformerConfig,
    ) -> std::result::Result<Self, LoaderError> {
        let vb = load_transformer_var_builder(layout, device, dtype)?;
        Self::from_var_builder(config.clone(), vb, device).map_err(LoaderError::Candle)
    }

    pub fn config(&self) -> &WanTransformerConfig {
        &self.config
    }

    /// Denoise forward: latents `[B,C,F,H,W]`, timestep `[B]`, text `[B,seq,text_dim]`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let (b, _c, num_frames, height, width) = hidden_states.dims5()?;
        let [p_t, p_h, p_w] = self.config.patch_size;
        let post_patch_frames = num_frames / p_t;
        let post_patch_height = height / p_h;
        let post_patch_width = width / p_w;
        let out_channels = self.config.out_channels;

        let rope = self.rope.forward(hidden_states)?;
        let mut hidden_states = self.patch_embedding.forward(hidden_states)?;

        let (temb, timestep_proj, encoder_hidden_states) = self
            .condition_embedder
            .forward(timestep, encoder_hidden_states)?;

        for block in &self.blocks {
            hidden_states = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &timestep_proj,
                &rope,
            )?;
        }

        let dtype = hidden_states.dtype();
        if hidden_states.device().is_cuda() {
            let table = self.scale_shift_table.to_dtype(dtype)?;
            let temb = temb.to_dtype(dtype)?.unsqueeze(1)?;
            let combined = table.broadcast_add(&temb)?;
            let shift = combined.narrow(1, 0, 1)?.squeeze(1)?;
            let scale = combined.narrow(1, 1, 1)?.squeeze(1)?;
            let norm_out = self.norm_out.forward(&hidden_states)?;
            let ones = Tensor::ones_like(&norm_out)?;
            let norm_out = norm_out.broadcast_mul(&(scale.broadcast_add(&ones)?))?;
            let norm_out = norm_out.broadcast_add(&shift)?;
            hidden_states = self.proj_out.forward(&norm_out)?;
        } else {
            let table = self.scale_shift_table.to_dtype(DType::F32)?;
            let temb_f32 = temb.to_dtype(DType::F32)?.unsqueeze(1)?;
            let combined = table.broadcast_add(&temb_f32)?;
            let shift = combined.narrow(1, 0, 1)?.squeeze(1)?;
            let scale = combined.narrow(1, 1, 1)?.squeeze(1)?;

            let hs_f32 = hidden_states.to_dtype(DType::F32)?;
            let norm_out = self.norm_out.forward(&hs_f32)?;
            let ones = Tensor::ones_like(&norm_out)?;
            let norm_out = norm_out.broadcast_mul(&(scale.broadcast_add(&ones)?))?;
            let norm_out = norm_out.broadcast_add(&shift)?;
            hidden_states = self
                .proj_out
                .forward(&norm_out.to_dtype(hidden_states.dtype())?)?;
        }

        let out_per_patch = out_channels * p_t * p_h * p_w;
        hidden_states = hidden_states.reshape(&[
            b,
            post_patch_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            out_channels,
        ])?;
        hidden_states = hidden_states.permute(vec![0, 7, 1, 4, 2, 5, 3, 6])?;
        hidden_states = hidden_states.reshape((
            b,
            out_channels,
            post_patch_frames * p_t,
            post_patch_height * p_h,
            post_patch_width * p_w,
        ))?;

        let _ = (self.inner_dim, out_per_patch);
        Ok(hidden_states)
    }
}
