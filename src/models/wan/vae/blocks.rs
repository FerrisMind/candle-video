//! Wan VAE residual, attention, and up blocks.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Module, VarBuilder};

use super::causal_conv3d::{FeatCache, WanCausalConv3d};
use super::resample::{ResampleMode, WanResample};
use super::rms_norm::WanRmsNorm;

#[derive(Debug)]
pub struct WanResidualBlock {
    norm1: WanRmsNorm,
    conv1: WanCausalConv3d,
    norm2: WanRmsNorm,
    conv2: WanCausalConv3d,
    conv_shortcut: Option<WanCausalConv3d>,
}

impl WanResidualBlock {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_shortcut = if in_dim != out_dim {
            Some(WanCausalConv3d::new(
                in_dim,
                out_dim,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: WanRmsNorm::new(in_dim, true, false, vb.pp("norm1"))?,
            conv1: WanCausalConv3d::new(
                in_dim,
                out_dim,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                vb.pp("conv1"),
            )?,
            norm2: WanRmsNorm::new(out_dim, true, false, vb.pp("norm2"))?,
            conv2: WanCausalConv3d::new(
                out_dim,
                out_dim,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                vb.pp("conv2"),
            )?,
            conv_shortcut,
        })
    }

    pub fn count_causal_conv3d(&self) -> usize {
        2 + usize::from(self.conv_shortcut.is_some())
    }

    pub fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let h = if let Some(shortcut) = &self.conv_shortcut {
            shortcut.forward(x, None)?
        } else {
            x.clone()
        };

        let mut x = self.norm1.forward(x)?;
        x = candle_nn::ops::silu(&x)?;
        x = self.conv1.forward_cached(&x, cache, true)?;

        x = self.norm2.forward(&x)?;
        x = candle_nn::ops::silu(&x)?;
        x = self.conv2.forward_cached(&x, cache, true)?;

        x.add(&h)
    }

}

#[derive(Debug)]
pub struct WanAttentionBlock {
    norm: WanRmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl WanAttentionBlock {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: WanRmsNorm::new(dim, true, true, vb.pp("norm"))?,
            to_qkv: candle_nn::conv2d(dim, dim * 3, 1, Default::default(), vb.pp("to_qkv"))?,
            proj: candle_nn::conv2d(dim, dim, 1, Default::default(), vb.pp("proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let (b, c, t, h, w) = x.dims5()?;
        let x = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
        let x = self.norm.forward(&x)?;
        let qkv = self.to_qkv.forward(&x)?;
        // `(bt, 1, hw, c*3)` after reshape + permute(0, 1, 3, 2)
        let qkv = qkv
            .reshape((b * t, 1, c * 3, h * w))?
            .permute((0, 1, 3, 2))?
            .contiguous()?;
        let q = qkv.narrow(3, 0, c)?;
        let k = qkv.narrow(3, c, c)?;
        let v = qkv.narrow(3, 2 * c, c)?;
        let scale = (c as f64).powf(-0.5);
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn = candle_nn::ops::softmax(
            &q.contiguous()?.matmul(&k_t)?.affine(scale, 0.0)?,
            3,
        )?;
        // `(bt, 1, hw, c)` -> `(bt, c, h, w)`
        let x = attn
            .matmul(&v.contiguous()?)?
            .squeeze(1)?
            .transpose(1, 2)?
            .reshape((b * t, c, h, w))?;
        let x = self.proj.forward(&x)?;
        let x = x.reshape((b, t, c, h, w))?.permute((0, 2, 1, 3, 4))?;
        x.add(&identity)
    }
}

#[derive(Debug)]
pub struct WanMidBlock {
    resnets: Vec<WanResidualBlock>,
    attentions: Vec<WanAttentionBlock>,
}

impl WanMidBlock {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            resnets: vec![
                WanResidualBlock::new(dim, dim, vb.pp("resnets").pp("0"))?,
                WanResidualBlock::new(dim, dim, vb.pp("resnets").pp("1"))?,
            ],
            attentions: vec![WanAttentionBlock::new(dim, vb.pp("attentions").pp("0"))?],
        })
    }

    pub fn count_causal_conv3d(&self) -> usize {
        self.resnets
            .iter()
            .map(WanResidualBlock::count_causal_conv3d)
            .sum()
    }

    pub fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let mut x = self.resnets[0].forward(x, cache)?;
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            x = attn.forward(&x)?;
            x = resnet.forward(&x, cache)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
pub struct WanUpBlock {
    resnets: Vec<WanResidualBlock>,
    upsampler: Option<WanResample>,
}

impl WanUpBlock {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        upsample_mode: Option<ResampleMode>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks + 1);
        let mut current = in_dim;
        for i in 0..=num_res_blocks {
            resnets.push(WanResidualBlock::new(
                current,
                out_dim,
                vb.pp("resnets").pp(i.to_string()),
            )?);
            current = out_dim;
        }
        let upsampler = upsample_mode
            .map(|mode| WanResample::new(out_dim, mode, vb.pp("upsamplers").pp("0")))
            .transpose()?;
        Ok(Self { resnets, upsampler })
    }

    pub fn count_causal_conv3d(&self) -> usize {
        let res: usize = self
            .resnets
            .iter()
            .map(WanResidualBlock::count_causal_conv3d)
            .sum();
        res + self
            .upsampler
            .as_ref()
            .map(WanResample::count_causal_conv3d)
            .unwrap_or(0)
    }

    pub fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x, cache)?;
        }
        if let Some(upsampler) = &self.upsampler {
            x = upsampler.forward(&x, cache)?;
        }
        Ok(x)
    }
}
