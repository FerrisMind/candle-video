//! UNet Spatio-Temporal Condition Model for SVD
//!
//! Main UNet architecture for Stable Video Diffusion.

use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, conv2d, linear};

use super::blocks::{
    CrossAttnDownBlockSpatioTemporal, CrossAttnUpBlockSpatioTemporal, DownBlockSpatioTemporal,
    UNetMidBlockSpatioTemporal, UpBlockSpatioTemporal,
};
use crate::svd::config::SvdUnetConfig;

/// Sinusoidal timestep embeddings
pub fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> Result<Tensor> {
    let dtype = timesteps.dtype();
    let half_dim = embedding_dim / 2;
    let exponent =
        Tensor::arange(0f32, half_dim as f32, timesteps.device())?.to_dtype(DType::F32)?;
    let exponent = (exponent * (-f64::ln(10000.0) / (half_dim as f64 - 1.0)))?;

    let timesteps = timesteps.to_dtype(DType::F32)?;
    let timesteps = timesteps.unsqueeze(1)?;
    let exponent = exponent.unsqueeze(0)?;

    let emb = timesteps.broadcast_mul(&exponent)?;
    let emb_cos = emb.cos()?;
    let emb_sin = emb.sin()?;

    Tensor::cat(&[emb_cos, emb_sin], 1)?.to_dtype(dtype)
}

/// Timestep embedding module (Linear -> SiLU -> Linear)
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(vb: VarBuilder, in_channels: usize, time_embed_dim: usize) -> Result<Self> {
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let sample = self.linear_1.forward(sample)?;
        let sample = candle_nn::ops::silu(&sample)?;
        self.linear_2.forward(&sample)
    }
}

/// Timesteps module for projecting scalar values to embeddings
#[derive(Debug)]
pub struct Timesteps {
    num_channels: usize,
}

impl Timesteps {
    pub fn new(num_channels: usize) -> Self {
        Self { num_channels }
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        get_timestep_embedding(timesteps, self.num_channels)
    }
}

/// Add time embedding for conditioning (fps, motion_bucket_id, noise_aug_strength)
#[derive(Debug)]
pub struct AddTimeEmbedding {
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
}

impl AddTimeEmbedding {
    pub fn new(
        vb: VarBuilder,
        addition_time_embed_dim: usize,
        time_embed_dim: usize,
    ) -> Result<Self> {
        let time_proj = Timesteps::new(addition_time_embed_dim);
        let time_embedding = TimestepEmbedding::new(
            vb.pp("add_embedding"),
            addition_time_embed_dim * 3, // fps + motion + noise_aug
            time_embed_dim,
        )?;
        Ok(Self {
            time_proj,
            time_embedding,
        })
    }

    pub fn forward(
        &self,
        fps: &Tensor,
        motion_bucket_id: &Tensor,
        noise_aug_strength: &Tensor,
    ) -> Result<Tensor> {
        let fps_embed = self.time_proj.forward(fps)?;
        let motion_embed = self.time_proj.forward(motion_bucket_id)?;
        let noise_aug_embed = self.time_proj.forward(noise_aug_strength)?;

        let time_embeds = Tensor::cat(&[fps_embed, motion_embed, noise_aug_embed], 1)?;
        self.time_embedding.forward(&time_embeds)
    }
}

/// Full UNet Spatio-Temporal Condition Model
#[derive(Debug)]
pub struct UNetSpatioTemporalConditionModel {
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    add_time_proj: Timesteps,
    add_embedding: TimestepEmbedding,

    down_blocks: Vec<DownBlock>,
    mid_block: UNetMidBlockSpatioTemporal,
    up_blocks: Vec<UpBlock>,

    conv_norm_out: candle_nn::GroupNorm,
    conv_out: Conv2d,
}

/// Enum for down blocks (3 has attention, last one doesn't)
#[derive(Debug)]
enum DownBlock {
    Standard(DownBlockSpatioTemporal),
    CrossAttn(CrossAttnDownBlockSpatioTemporal),
}

/// Enum for up blocks (first one has no attention)
#[derive(Debug)]
enum UpBlock {
    Standard(UpBlockSpatioTemporal),
    CrossAttn(CrossAttnUpBlockSpatioTemporal),
}

impl UNetSpatioTemporalConditionModel {
    pub fn new(vb: VarBuilder, config: &SvdUnetConfig) -> Result<Self> {
        // Input convolution
        let conv_in = conv2d(
            config.in_channels,
            config.block_out_channels[0],
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_in"),
        )?;

        // Time embedding
        let time_proj = Timesteps::new(config.block_out_channels[0]);
        let time_embed_dim = config.block_out_channels[0] * 4;
        let time_embedding = TimestepEmbedding::new(
            vb.pp("time_embedding"),
            config.block_out_channels[0],
            time_embed_dim,
        )?;

        // Add time projection (for fps, motion_bucket_id, noise_aug_strength)
        let add_time_proj = Timesteps::new(config.addition_time_embed_dim);
        let add_embedding = TimestepEmbedding::new(
            vb.pp("add_embedding"),
            config.addition_time_embed_dim * 3,
            time_embed_dim,
        )?;

        // Build down blocks - last one has NO attention
        let mut down_blocks = Vec::new();
        let mut output_channel = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = out_ch;
            let is_final = i == config.block_out_channels.len() - 1;

            if is_final {
                // Last down_block has NO attention
                down_blocks.push(DownBlock::Standard(DownBlockSpatioTemporal::new(
                    vb.pp("down_blocks").pp(i),
                    input_channel,
                    output_channel,
                    config.layers_per_block,
                    Some(time_embed_dim),
                    false, // No downsampler for last block
                )?));
            } else {
                down_blocks.push(DownBlock::CrossAttn(CrossAttnDownBlockSpatioTemporal::new(
                    vb.pp("down_blocks").pp(i),
                    input_channel,
                    output_channel,
                    config.layers_per_block,
                    config.num_attention_heads.get(i).copied().unwrap_or(8),
                    config.cross_attention_dim,
                    Some(time_embed_dim),
                    true, // Has downsampler
                )?));
            }
        }

        // Mid block
        let mid_block = UNetMidBlockSpatioTemporal::new(
            vb.pp("mid_block"),
            *config.block_out_channels.last().unwrap(),
            Some(time_embed_dim),
            config.num_attention_heads.last().copied().unwrap_or(8),
            config.cross_attention_dim,
        )?;

        // Build up blocks - exact input channels from diffusers weights analysis
        // up_blocks.0: in=[2560,2560,2560], out=1280 (no attention)
        // up_blocks.1: in=[2560,2560,1920], out=1280 (attention)
        // up_blocks.2: in=[1920,1280,960], out=640 (attention)
        // up_blocks.3: in=[960,640,640], out=320 (attention)
        let up_block_configs: Vec<(Vec<usize>, usize, bool, bool)> = vec![
            (vec![2560, 2560, 2560], 1280, false, true), // no attn, has upsample
            (vec![2560, 2560, 1920], 1280, true, true),  // has attn, has upsample
            (vec![1920, 1280, 960], 640, true, true),    // has attn, has upsample
            (vec![960, 640, 640], 320, true, false),     // has attn, no upsample (final)
        ];

        let mut up_blocks = Vec::new();
        for (i, (in_channels_list, out_ch, has_attention, add_upsample)) in
            up_block_configs.into_iter().enumerate()
        {
            if has_attention {
                up_blocks.push(UpBlock::CrossAttn(CrossAttnUpBlockSpatioTemporal::new(
                    vb.pp("up_blocks").pp(i),
                    &in_channels_list,
                    out_ch,
                    config.num_attention_heads.get(i).copied().unwrap_or(8),
                    config.cross_attention_dim,
                    Some(time_embed_dim),
                    add_upsample,
                )?));
            } else {
                up_blocks.push(UpBlock::Standard(UpBlockSpatioTemporal::new(
                    vb.pp("up_blocks").pp(i),
                    &in_channels_list,
                    out_ch,
                    Some(time_embed_dim),
                    add_upsample,
                )?));
            }
        }

        // Output
        let conv_norm_out = candle_nn::group_norm(
            32,
            config.block_out_channels[0],
            1e-6,
            vb.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(
            config.block_out_channels[0],
            config.out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        added_time_ids: &Tensor,
        num_frames: usize,
        image_only_indicator: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Time embedding
        let t_emb = self.time_proj.forward(timestep)?;
        let t_emb = self.time_embedding.forward(&t_emb)?;

        // 2. Additional time embeddings (fps, motion_bucket_id, noise_aug_strength)
        // added_time_ids: [B, 3] where dims are [fps, motion_bucket_id, noise_aug_strength]
        let fps = added_time_ids.i((.., 0))?;
        let motion = added_time_ids.i((.., 1))?;
        let noise_aug = added_time_ids.i((.., 2))?;

        let fps_emb = self.add_time_proj.forward(&fps)?;
        let motion_emb = self.add_time_proj.forward(&motion)?;
        let noise_aug_emb = self.add_time_proj.forward(&noise_aug)?;

        let aug_emb = Tensor::cat(&[fps_emb, motion_emb, noise_aug_emb], 1)?;
        let aug_emb = self.add_embedding.forward(&aug_emb)?;

        // Combined time embedding
        let emb = (t_emb + aug_emb)?;

        // 3. Pre-process
        let sample = self.conv_in.forward(sample)?;

        // 4. Down blocks
        let mut down_block_res_samples = vec![sample.clone()];
        let mut sample = sample;

        for down_block in &self.down_blocks {
            let (h, res_samples) = match down_block {
                DownBlock::Standard(block) => {
                    block.forward(&sample, Some(&emb), image_only_indicator, num_frames)?
                }
                DownBlock::CrossAttn(block) => block.forward(
                    &sample,
                    Some(&emb),
                    Some(encoder_hidden_states),
                    image_only_indicator,
                    num_frames,
                )?,
            };
            sample = h;
            down_block_res_samples.extend(res_samples);
        }

        // 5. Mid block
        sample = self.mid_block.forward(
            &sample,
            Some(&emb),
            Some(encoder_hidden_states),
            image_only_indicator,
            num_frames,
        )?;

        // 6. Up blocks
        for up_block in &self.up_blocks {
            sample = match up_block {
                UpBlock::Standard(block) => block.forward(
                    &sample,
                    &mut down_block_res_samples,
                    Some(&emb),
                    image_only_indicator,
                    num_frames,
                )?,
                UpBlock::CrossAttn(block) => block.forward(
                    &sample,
                    &mut down_block_res_samples,
                    Some(&emb),
                    Some(encoder_hidden_states),
                    image_only_indicator,
                    num_frames,
                )?,
            };
        }

        // 7. Post-process
        let sample = self.conv_norm_out.forward(&sample)?;
        let sample = candle_nn::ops::silu(&sample)?;
        self.conv_out.forward(&sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_embedding() {
        let device = candle_core::Device::Cpu;
        let timesteps = Tensor::new(&[1.0f32, 10.0, 100.0], &device).unwrap();
        let emb = get_timestep_embedding(&timesteps, 64).unwrap();
        assert_eq!(emb.dims(), &[3, 64]);
    }
}
