//! Spatio-Temporal ResNet blocks for SVD UNet
//!
//! These blocks extend standard ResNet blocks with temporal mixing capabilities.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

/// Standard ResNet 2D block (matches diffusers format)
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

        // Add time embedding if present
        if let (Some(proj), Some(temb)) = (&self.time_emb_proj, temb) {
            let temb_out = candle_nn::ops::silu(temb)?;
            let temb_out = proj.forward(&temb_out)?;
            // temb_out: [B, C] -> [B, C, 1, 1]
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

/// Conv3D for temporal convolutions - weights shape: [out, in, T, H, W]
/// For SVD temporal blocks: T=3, H=1, W=1
#[derive(Debug)]
pub struct TemporalConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
}

impl TemporalConv3d {
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        // Weight shape: [out_ch, in_ch, 3, 1, 1] for temporal conv
        let weight = vb.get((out_channels, in_channels, 3, 1, 1), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();

        Ok(Self {
            weight,
            bias,
            padding: 1, // padding=1 for kernel_size=3
        })
    }

    pub fn forward(&self, x: &Tensor, num_frames: usize) -> Result<Tensor> {
        // x: [B*T, C, H, W]
        let (batch_frames, c, h, w) = x.dims4()?;
        let batch_size = batch_frames / num_frames;

        // Weight: [out_ch, in_ch, 3, 1, 1]
        let (out_ch, _in_ch, _kt, kh, kw) = self.weight.dims5()?;

        if kh == 1 && kw == 1 {
            // Temporal-only convolution: reshape and use conv1d
            // Reshape: [B*T, C, H, W] -> [B, C, T, H, W] -> [B*H*W, C, T]
            let x = x.reshape((batch_size, num_frames, c, h, w))?;
            let x = x.permute((0, 3, 4, 2, 1))?; // [B, H, W, C, T]
            let x = x.reshape((batch_size * h * w, c, num_frames))?;

            // Squeeze weight to [out_ch, in_ch, kt]
            let weight_1d = self.weight.squeeze(4)?.squeeze(3)?;

            // Apply conv1d
            let out = x.conv1d(&weight_1d, self.padding, 1, 1, 1)?;

            // Add bias
            let out = if let Some(bias) = &self.bias {
                out.broadcast_add(&bias.reshape((1, out_ch, 1))?)?
            } else {
                out
            };

            // Reshape back: [B*H*W, out_ch, T] -> [B*T, out_ch, H, W]
            let out = out.reshape((batch_size, h, w, out_ch, num_frames))?;
            let out = out.permute((0, 4, 3, 1, 2))?; // [B, T, C, H, W]
            out.reshape((batch_frames, out_ch, h, w))
        } else {
            // Fallback: identity
            Ok(x.clone())
        }
    }
}

/// Temporal ResNet block using Conv3D (kernel 3x1x1 for temporal mixing)
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

        // Norm + SiLU + Conv1
        let mut h = self.norm1.forward(x)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv1.forward(&h, num_frames)?;

        // Add time embedding
        if let (Some(proj), Some(temb)) = (&self.time_emb_proj, temb) {
            let temb_out = candle_nn::ops::silu(temb)?;
            let temb_out = proj.forward(&temb_out)?;
            let temb_out = temb_out.unsqueeze(2)?.unsqueeze(3)?;
            h = h.broadcast_add(&temb_out)?;
        }

        // Norm + SiLU + Conv2
        h = self.norm2.forward(&h)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv2.forward(&h, num_frames)?;

        // Skip connection (identity for same channels)
        h + residual
    }
}

/// Alpha blender for merging spatial and temporal features
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
        // diffusers formula: alpha * spatial + (1-alpha) * temporal
        (x_spatial * &alpha)? + (x_temporal * one_minus_alpha)?
    }
}

/// Spatio-Temporal ResNet block combining spatial and temporal processing
/// Matches diffusers structure: spatial_res_block, temporal_res_block, time_mixer
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
        // Spatial processing
        let h_spatial = self.spatial_res_block.forward(x, temb)?;

        // Temporal processing
        let h_temporal = self
            .temporal_res_block
            .forward(&h_spatial, temb, num_frames)?;

        // Blend spatial and temporal
        self.time_mixer
            .forward(&h_spatial, &h_temporal, image_only_indicator)
    }
}
