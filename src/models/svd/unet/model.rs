use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

use super::super::config::SvdUnetConfig;
use super::blocks::{
    CrossAttnDownBlockSpatioTemporal, CrossAttnUpBlockSpatioTemporal, DownBlockSpatioTemporal,
    UNetMidBlockSpatioTemporal, UpBlockSpatioTemporal,
};
use crate::interfaces::embeddings::{TimestepEmbedding, get_timestep_embedding};

fn debug_check_tensor(name: &str, tensor: &Tensor) {
    if std::env::var("DEBUG_UNET").is_ok()
        && let Ok(f32_tensor) = tensor.to_dtype(DType::F32)
        && let Ok(flat) = f32_tensor.flatten_all()
    {
        let has_nan = flat
            .to_vec1::<f32>()
            .map(|v| v.iter().any(|x| x.is_nan()))
            .unwrap_or(false);
        let has_inf = flat
            .to_vec1::<f32>()
            .map(|v| v.iter().any(|x| x.is_infinite()))
            .unwrap_or(false);
        let min = flat.min(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
        let max = flat.max(0).ok().and_then(|t| t.to_scalar::<f32>().ok());
        if has_nan || has_inf {
            println!(
                "    [UNET] {} has NaN={}, Inf={}, min={:?}, max={:?}",
                name, has_nan, has_inf, min, max
            );
        }
    }
}

#[derive(Debug)]
pub struct Timesteps {
    num_channels: usize,
}

impl Timesteps {
    pub fn new(num_channels: usize) -> Self {
        Self { num_channels }
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        get_timestep_embedding(timesteps, self.num_channels, true)
    }
}

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
            addition_time_embed_dim * 3,
            time_embed_dim,
            vb.pp("add_embedding"),
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

#[derive(Debug)]
enum DownBlock {
    Standard(DownBlockSpatioTemporal),
    CrossAttn(CrossAttnDownBlockSpatioTemporal),
}

#[derive(Debug)]
enum UpBlock {
    Standard(UpBlockSpatioTemporal),
    CrossAttn(CrossAttnUpBlockSpatioTemporal),
}

impl UNetSpatioTemporalConditionModel {
    pub fn new(vb: VarBuilder, config: &SvdUnetConfig) -> Result<Self> {
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

        let time_proj = Timesteps::new(config.block_out_channels[0]);
        let time_embed_dim = config.block_out_channels[0] * 4;
        let time_embedding = TimestepEmbedding::new(
            config.block_out_channels[0],
            time_embed_dim,
            vb.pp("time_embedding"),
        )?;

        let add_time_proj = Timesteps::new(config.addition_time_embed_dim);
        let add_embedding = TimestepEmbedding::new(
            config.addition_time_embed_dim * 3,
            time_embed_dim,
            vb.pp("add_embedding"),
        )?;

        let mut down_blocks = Vec::new();
        let mut output_channel = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = out_ch;
            let is_final = i == config.block_out_channels.len() - 1;

            if is_final {
                down_blocks.push(DownBlock::Standard(DownBlockSpatioTemporal::new(
                    vb.pp("down_blocks").pp(i),
                    input_channel,
                    output_channel,
                    config.layers_per_block,
                    Some(time_embed_dim),
                    false,
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
                    true,
                )?));
            }
        }

        let mid_block = UNetMidBlockSpatioTemporal::new(
            vb.pp("mid_block"),
            *config.block_out_channels.last().unwrap(),
            Some(time_embed_dim),
            config.num_attention_heads.last().copied().unwrap_or(8),
            config.cross_attention_dim,
        )?;

        let up_block_configs: Vec<(Vec<usize>, usize, bool, bool)> = vec![
            (vec![2560, 2560, 2560], 1280, false, true),
            (vec![2560, 2560, 1920], 1280, true, true),
            (vec![1920, 1280, 960], 640, true, true),
            (vec![960, 640, 640], 320, true, false),
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
        debug_check_tensor("INPUT sample", sample);
        debug_check_tensor("INPUT encoder_hidden_states", encoder_hidden_states);
        debug_check_tensor("INPUT added_time_ids", added_time_ids);

        let t_emb = self.time_proj.forward(timestep)?;
        let t_emb = self.time_embedding.forward(&t_emb)?;
        debug_check_tensor("t_emb", &t_emb);

        let fps = added_time_ids.i((.., 0))?;
        let motion = added_time_ids.i((.., 1))?;
        let noise_aug = added_time_ids.i((.., 2))?;

        let fps_emb = self.add_time_proj.forward(&fps)?;
        let motion_emb = self.add_time_proj.forward(&motion)?;
        let noise_aug_emb = self.add_time_proj.forward(&noise_aug)?;

        let aug_emb = Tensor::cat(&[fps_emb, motion_emb, noise_aug_emb], 1)?;
        let aug_emb = self.add_embedding.forward(&aug_emb)?;
        debug_check_tensor("aug_emb", &aug_emb);

        let emb = (t_emb + aug_emb)?;
        debug_check_tensor("combined_emb", &emb);

        let sample = self.conv_in.forward(sample)?;
        debug_check_tensor("after_conv_in", &sample);

        let mut down_block_res_samples = vec![sample.clone()];
        let mut sample = sample;

        for (idx, down_block) in self.down_blocks.iter().enumerate() {
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
            debug_check_tensor(&format!("after_down_block[{}]", idx), &sample);
        }

        sample = self.mid_block.forward(
            &sample,
            Some(&emb),
            Some(encoder_hidden_states),
            image_only_indicator,
            num_frames,
        )?;
        debug_check_tensor("after_mid_block", &sample);

        for (idx, up_block) in self.up_blocks.iter().enumerate() {
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
            debug_check_tensor(&format!("after_up_block[{}]", idx), &sample);
        }

        let sample = self.conv_norm_out.forward(&sample)?;
        let sample = candle_nn::ops::silu(&sample)?;
        debug_check_tensor("after_conv_norm_out_silu", &sample);

        let output = self.conv_out.forward(&sample)?;
        debug_check_tensor("after_conv_out", &output);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_embedding() {
        let device = candle_core::Device::Cpu;
        let timesteps = Tensor::new(&[1.0f32, 10.0, 100.0], &device).unwrap();
        let emb = get_timestep_embedding(&timesteps, 64, true).unwrap();
        assert_eq!(emb.dims(), &[3, 64]);
    }
}
