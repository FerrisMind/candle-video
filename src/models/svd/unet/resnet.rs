use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

use crate::ops::conv3d::{Conv3d, Conv3dConfig};

#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: candle_nn::GroupNorm,
    conv1: Conv2d,
    norm2: candle_nn::GroupNorm,
    conv2: Conv2d,
    time_emb_proj: Option<candle_nn::Linear>,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock2D {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
    ) -> Result<Self> {
        let norm1 = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(
            in_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let norm2 = candle_nn::group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(
            out_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        let time_emb_proj = if let Some(temb_ch) = temb_channels {
            Some(candle_nn::linear(
                temb_ch,
                out_channels,
                vb.pp("time_emb_proj"),
            )?)
        } else {
            None
        };

        let conv_shortcut = if in_channels != out_channels {
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                Default::default(),
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
            conv_shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;

        let mut h = self.norm1.forward(x)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv1.forward(&h)?;

        if let (Some(proj), Some(temb)) = (&self.time_emb_proj, temb) {
            let temb_out = candle_nn::ops::silu(temb)?;
            let temb_out = proj.forward(&temb_out)?;

            let temb_out = temb_out.unsqueeze(2)?.unsqueeze(3)?;
            h = h.broadcast_add(&temb_out)?;
        }

        h = self.norm2.forward(&h)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv2.forward(&h)?;

        let residual = if let Some(conv) = &self.conv_shortcut {
            conv.forward(residual)?
        } else {
            residual.clone()
        };

        h + residual
    }
}

#[derive(Debug, Clone)]
pub struct TemporalConv3d {
    conv: Conv3d,
}

impl TemporalConv3d {
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let config = Conv3dConfig::new((3, 1, 1))
            .with_padding((1, 0, 0))
            .with_stride((1, 1, 1));

        let conv = Conv3d::new(in_channels, out_channels, config, vb)?;

        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor, num_frames: usize) -> Result<Tensor> {
        let (batch_frames, c, h, w) = x.dims4()?;
        let batch_size = batch_frames / num_frames;

        let x_5d = x.reshape((batch_size, num_frames, c, h, w))?;
        let x_5d = x_5d.permute((0, 2, 1, 3, 4))?;

        let out_5d = self.conv.forward(&x_5d)?;

        let (b, out_c, t, h_out, w_out) = out_5d.dims5()?;
        let out_4d = out_5d.permute((0, 2, 1, 3, 4))?;
        out_4d.reshape((b * t, out_c, h_out, w_out))
    }
}

#[derive(Debug)]
pub struct TemporalResnetBlock {
    norm1: candle_nn::GroupNorm,
    conv1: TemporalConv3d,
    norm2: candle_nn::GroupNorm,
    conv2: TemporalConv3d,
    time_emb_proj: Option<candle_nn::Linear>,
}

impl TemporalResnetBlock {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
    ) -> Result<Self> {
        let norm1 = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = TemporalConv3d::new(vb.pp("conv1"), in_channels, out_channels)?;
        let norm2 = candle_nn::group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = TemporalConv3d::new(vb.pp("conv2"), out_channels, out_channels)?;

        let time_emb_proj = if let Some(temb_ch) = temb_channels {
            Some(candle_nn::linear(
                temb_ch,
                out_channels,
                vb.pp("time_emb_proj"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, num_frames: usize) -> Result<Tensor> {
        let residual = x;

        let mut h = self.norm1.forward(x)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv1.forward(&h, num_frames)?;

        if let (Some(proj), Some(temb)) = (&self.time_emb_proj, temb) {
            let temb_out = candle_nn::ops::silu(temb)?;
            let temb_out = proj.forward(&temb_out)?;
            let temb_out = temb_out.unsqueeze(2)?.unsqueeze(3)?;
            h = h.broadcast_add(&temb_out)?;
        }

        h = self.norm2.forward(&h)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv2.forward(&h, num_frames)?;

        h + residual
    }
}

#[derive(Debug)]
pub struct AlphaBlender {
    mix_factor: Tensor,
    merge_strategy: MergeStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    Learned,
    LearnedWithImages,
    Fixed,
}

impl AlphaBlender {
    pub fn new(vb: VarBuilder, merge_strategy: MergeStrategy) -> Result<Self> {
        let mix_factor = vb.get(1, "mix_factor")?;
        Ok(Self {
            mix_factor,
            merge_strategy,
        })
    }

    pub fn forward(
        &self,
        x_spatial: &Tensor,
        x_temporal: &Tensor,
        _image_only_indicator: Option<&Tensor>,
    ) -> Result<Tensor> {
        let alpha = match self.merge_strategy {
            MergeStrategy::Learned | MergeStrategy::LearnedWithImages => {
                candle_nn::ops::sigmoid(&self.mix_factor)?
            }
            MergeStrategy::Fixed => self.mix_factor.clone(),
        };

        let alpha = alpha.broadcast_as(x_spatial.shape())?;
        let one_minus_alpha = (1.0 - &alpha)?;

        (x_spatial * &alpha)? + (x_temporal * one_minus_alpha)?
    }
}

#[derive(Debug)]
pub struct SpatioTemporalResBlock {
    spatial_res_block: ResnetBlock2D,
    temporal_res_block: TemporalResnetBlock,
    time_mixer: AlphaBlender,
}

impl SpatioTemporalResBlock {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        temb_channels: Option<usize>,
    ) -> Result<Self> {
        let spatial_res_block = ResnetBlock2D::new(
            vb.pp("spatial_res_block"),
            in_channels,
            out_channels,
            temb_channels,
        )?;
        let temporal_res_block = TemporalResnetBlock::new(
            vb.pp("temporal_res_block"),
            out_channels,
            out_channels,
            temb_channels,
        )?;
        let time_mixer = AlphaBlender::new(vb.pp("time_mixer"), MergeStrategy::Learned)?;

        Ok(Self {
            spatial_res_block,
            temporal_res_block,
            time_mixer,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        temb: Option<&Tensor>,
        image_only_indicator: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<Tensor> {
        let h_spatial = self.spatial_res_block.forward(x, temb)?;

        let h_temporal = self
            .temporal_res_block
            .forward(&h_spatial, temb, num_frames)?;

        self.time_mixer
            .forward(&h_spatial, &h_temporal, image_only_indicator)
    }
}
