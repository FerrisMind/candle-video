//! Wan 2.1 `WanDecoder3d` (non-residual path).

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::blocks::{WanMidBlock, WanUpBlock};
use super::causal_conv3d::{FeatCache, WanCausalConv3d};
use super::resample::ResampleMode;
use super::rms_norm::WanRmsNorm;

use crate::models::wan::configs::WanVaeConfig;

#[derive(Debug)]
pub struct WanDecoder3d {
    conv_in: WanCausalConv3d,
    mid_block: WanMidBlock,
    up_blocks: Vec<WanUpBlock>,
    norm_out: WanRmsNorm,
    conv_out: WanCausalConv3d,
}

impl WanDecoder3d {
    pub fn new(cfg: &WanVaeConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.base_dim;
        let dim_mult = &cfg.dim_mult;
        let temperal_upsample: Vec<bool> = cfg.temporal_downsample.iter().rev().copied().collect();

        let dims: Vec<usize> = [dim_mult[dim_mult.len() - 1]]
            .into_iter()
            .chain(dim_mult.iter().rev().copied())
            .map(|u| dim * u)
            .collect();

        let mut up_blocks = Vec::new();
        for (i, (&in_d, &out_d)) in dims.iter().zip(dims.iter().skip(1)).enumerate() {
            let mut in_dim = in_d;
            if i > 0 {
                in_dim /= 2;
            }
            let up_flag = i != dim_mult.len() - 1;
            let upsample_mode = if up_flag {
                if temperal_upsample[i] {
                    Some(ResampleMode::Upsample3d)
                } else {
                    Some(ResampleMode::Upsample2d)
                }
            } else {
                None
            };
            up_blocks.push(WanUpBlock::new(
                in_dim,
                out_d,
                cfg.num_res_blocks,
                upsample_mode,
                vb.pp("up_blocks").pp(i.to_string()),
            )?);
        }

        let out_dim = *dims.last().ok_or_else(|| {
            candle_core::Error::Msg("Wan VAE decoder has no output channel level".into())
        })?;

        Ok(Self {
            conv_in: WanCausalConv3d::new(
                cfg.z_dim,
                dims[0],
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                vb.pp("conv_in"),
            )?,
            mid_block: WanMidBlock::new(dims[0], vb.pp("mid_block"))?,
            up_blocks,
            norm_out: WanRmsNorm::new(out_dim, true, false, vb.pp("norm_out"))?,
            conv_out: WanCausalConv3d::new(
                out_dim,
                3,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                vb.pp("conv_out"),
            )?,
        })
    }

    pub fn count_causal_conv3d(&self) -> usize {
        let mut n = 2;
        n += self.mid_block.count_causal_conv3d();
        n += self
            .up_blocks
            .iter()
            .map(WanUpBlock::count_causal_conv3d)
            .sum::<usize>();
        n
    }

    pub fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let mut x = self.conv_in.forward_cached(x, cache, true)?;
        x = self.mid_block.forward(&x, cache)?;
        for up_block in &self.up_blocks {
            x = up_block.forward(&x, cache)?;
        }
        x = self.norm_out.forward(&x)?;
        x = candle_nn::ops::silu(&x)?;
        self.conv_out.forward_cached(&x, cache, true)
    }
}
