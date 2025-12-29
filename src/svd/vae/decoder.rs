//! Temporal Decoder for SVD VAE
//!
//! Decoder with temporal awareness for producing temporally consistent video frames.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

use crate::svd::config::SvdVaeConfig;

/// Spatio-temporal residual block for temporal decoder
#[derive(Debug)]
struct SpatioTemporalResBlock {
    spatial_res_block: ResnetBlock2D,
    temporal_res_block: TemporalResnetBlock,
    time_mixer: AlphaBlender,
}

impl SpatioTemporalResBlock {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let spatial_res_block =
            ResnetBlock2D::new(vb.pp("spatial_res_block"), in_channels, out_channels)?;
        let temporal_res_block =
            TemporalResnetBlock::new(vb.pp("temporal_res_block"), out_channels, out_channels)?;
        let time_mixer = AlphaBlender::new(vb.pp("time_mixer"))?;

        Ok(Self {
            spatial_res_block,
            temporal_res_block,
            time_mixer,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        image_only_indicator: &Tensor,
        num_frames: usize,
    ) -> Result<Tensor> {
        let h = self.spatial_res_block.forward(x)?;
        let h_temporal = self.temporal_res_block.forward(&h, num_frames)?;
        self.time_mixer
            .forward(&h, &h_temporal, image_only_indicator)
    }
}

/// Standard ResnetBlock2D
#[derive(Debug)]
struct ResnetBlock2D {
    norm1: candle_nn::GroupNorm,
    conv1: Conv2d,
    norm2: candle_nn::GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock2D {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
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
            conv_shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;

        let h = self.norm1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        let residual = if let Some(conv) = &self.conv_shortcut {
            conv.forward(residual)?
        } else {
            residual.clone()
        };

        h + residual
    }
}

/// Temporal ResnetBlock for mixing across frames
#[derive(Debug)]
struct TemporalResnetBlock {
    norm1: candle_nn::GroupNorm,
    conv1: Conv3d,
    norm2: candle_nn::GroupNorm,
    conv2: Conv3d,
    in_channels: usize,
    out_channels: usize,
}

/// Simple Conv3d wrapper (using candle's conv2d repeated for temporal)
#[derive(Debug)]
struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: (usize, usize, usize),
    padding: (usize, usize, usize),
}

impl Conv3d {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel_size;
        let weight = vb.get((out_channels, in_channels, kt, kh, kw), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();

        Ok(Self {
            weight,
            bias,
            kernel_size,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T, H, W]
        let (_kt, kh, kw) = self.kernel_size;
        let (pt, _ph, _pw) = self.padding;

        // Pad temporal dimension
        let x = if pt > 0 {
            x.pad_with_zeros(2, pt, pt)?
        } else {
            x.clone()
        };

        // For kernel_size=(3,1,1), we can use conv1d for temporal
        if kh == 1 && kw == 1 {
            // Temporal-only convolution
            let x = x.permute((0, 3, 4, 1, 2))?; // [B, H, W, C, T]
            let (b, h, w, c, t) = x.dims5()?;
            let x = x.reshape((b * h * w, c, t))?;

            // Use conv1d for temporal
            let weight = self.weight.squeeze(3)?.squeeze(3)?; // [out, in, kt]
            let out = x.conv1d(&weight, 0, 1, 1, 1)?;

            if let Some(bias) = &self.bias {
                let out = out.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)?;
                let out = out.reshape((b, h, w, bias.dim(0)?, out.dim(2)?))?;
                out.permute((0, 3, 4, 1, 2))
            } else {
                let out_c = self.weight.dim(0)?;
                let out = out.reshape((b, h, w, out_c, out.dim(2)?))?;
                out.permute((0, 3, 4, 1, 2))
            }
        } else {
            // For complex kernels, identity fallback
            Ok(x)
        }
    }
}

impl TemporalResnetBlock {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let norm1 = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = Conv3d::new(
            vb.pp("conv1"),
            in_channels,
            out_channels,
            (3, 1, 1),
            (1, 0, 0),
        )?;
        let norm2 = candle_nn::group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = Conv3d::new(
            vb.pp("conv2"),
            out_channels,
            out_channels,
            (3, 1, 1),
            (1, 0, 0),
        )?;

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            in_channels,
            out_channels,
        })
    }

    fn forward(&self, x: &Tensor, num_frames: usize) -> Result<Tensor> {
        // x: [B*T, C, H, W] - reshape to [B, C, T, H, W] for temporal conv
        let (batch_frames, c, h, w) = x.dims4()?;
        let batch_size = batch_frames / num_frames;

        // Reshape to [B, T, C, H, W] then [B, C, T, H, W]
        let x_5d = x
            .reshape((batch_size, num_frames, c, h, w))?
            .permute((0, 2, 1, 3, 4))?; // [B, C, T, H, W]

        let residual = &x_5d;

        // Apply norm1 to each frame: need to reshape for GroupNorm
        // GroupNorm expects [N, C, ...], so flatten B*T dimension
        let h = x_5d
            .permute((0, 2, 1, 3, 4))? // [B, T, C, H, W]
            .reshape((batch_frames, c, h, w))?;
        let h = self.norm1.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;

        // Reshape back to 5D for conv3d
        let h = h
            .reshape((batch_size, num_frames, c, h.dim(2)?, h.dim(3)?))?
            .permute((0, 2, 1, 3, 4))?; // [B, C, T, H, W]
        let h = self.conv1.forward(&h)?;

        // Apply norm2
        let (_, c2, _, h2, w2) = h.dims5()?;
        let h = h
            .permute((0, 2, 1, 3, 4))? // [B, T, C, H, W]
            .reshape((batch_frames, c2, h2, w2))?;
        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;

        // Reshape back and apply conv2
        let h = h
            .reshape((batch_size, num_frames, c2, h2, w2))?
            .permute((0, 2, 1, 3, 4))?;
        let h = self.conv2.forward(&h)?;

        // Residual connection (identity if same channels)
        let out = if self.in_channels == self.out_channels {
            (h + residual)?
        } else {
            h
        };

        // Reshape back to [B*T, C, H, W]
        let (_, c_out, _, h_out, w_out) = out.dims5()?;
        out.permute((0, 2, 1, 3, 4))? // [B, T, C, H, W]
            .reshape((batch_frames, c_out, h_out, w_out))
    }
}

/// Alpha blender for mixing spatial and temporal features
#[derive(Debug)]
struct AlphaBlender {
    mix_factor: Tensor,
}

impl AlphaBlender {
    fn new(vb: VarBuilder) -> Result<Self> {
        let mix_factor = vb.get(1, "mix_factor")?;
        Ok(Self { mix_factor })
    }

    fn forward(
        &self,
        x_spatial: &Tensor,
        x_temporal: &Tensor,
        _image_only_indicator: &Tensor,
    ) -> Result<Tensor> {
        // diffusers formula with switch_spatial_to_temporal_mix=True (for VAE decoder):
        // alpha = 1.0 - alpha, then: alpha * spatial + (1-alpha) * temporal
        // Simplified: (1-alpha) * spatial + alpha * temporal
        let alpha = candle_nn::ops::sigmoid(&self.mix_factor)?;
        let alpha = alpha.broadcast_as(x_spatial.shape())?;

        // With switch_spatial_to_temporal_mix=True, we swap the roles
        let one_minus_alpha = (1.0 - &alpha)?;
        (x_spatial * one_minus_alpha)? + (x_temporal * &alpha)?
    }
}

/// Upsample block for decoder
#[derive(Debug)]
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let conv = conv2d(
            channels,
            channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&x)
    }
}

/// Up block for Temporal Decoder
#[derive(Debug)]
struct UpBlockTemporalDecoder {
    resnets: Vec<SpatioTemporalResBlock>,
    upsamplers: Option<Upsample2D>,
}

impl UpBlockTemporalDecoder {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_upsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(SpatioTemporalResBlock::new(
                vb.pp("resnets").pp(i),
                in_ch,
                out_channels,
            )?);
        }

        let upsamplers = if add_upsample {
            Some(Upsample2D::new(vb.pp("upsamplers").pp("0"), out_channels)?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            upsamplers,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        image_only_indicator: &Tensor,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h, image_only_indicator, num_frames)?;
        }
        if let Some(up) = &self.upsamplers {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// Attention block for decoder mid-block (matches diffusers)
#[derive(Debug)]
struct AttentionBlock {
    group_norm: candle_nn::GroupNorm,
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    channels: usize,
}

impl AttentionBlock {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let group_norm = candle_nn::group_norm(32, channels, 1e-6, vb.pp("group_norm"))?;
        let to_q = candle_nn::linear(channels, channels, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(channels, channels, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(channels, channels, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(channels, channels, vb.pp("to_out").pp("0"))?;
        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let residual = x;

        let x = self.group_norm.forward(x)?;
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?;

        let q = self.to_q.forward(&x)?;
        let k = self.to_k.forward(&x)?;
        let v = self.to_v.forward(&x)?;

        let scale = (self.channels as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        let out = self.to_out.forward(&out)?;
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;

        out + residual
    }
}

/// Mid block for Temporal Decoder (with self-attention like diffusers)
#[derive(Debug)]
struct MidBlockTemporalDecoder {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Vec<AttentionBlock>,
}

impl MidBlockTemporalDecoder {
    fn new(vb: VarBuilder, channels: usize, num_layers: usize) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::new();

        for i in 0..num_layers {
            resnets.push(SpatioTemporalResBlock::new(
                vb.pp("resnets").pp(i),
                channels,
                channels,
            )?);
        }

        // diffusers has 1 attention block applied after first resnet
        attentions.push(AttentionBlock::new(vb.pp("attentions").pp("0"), channels)?);

        Ok(Self { resnets, attentions })
    }

    fn forward(
        &self,
        x: &Tensor,
        image_only_indicator: &Tensor,
        num_frames: usize,
    ) -> Result<Tensor> {
        // Process first resnet
        let mut h = self.resnets[0].forward(x, image_only_indicator, num_frames)?;

        // Apply attention after first resnet, then remaining resnets
        for (resnet, attn) in self.resnets[1..].iter().zip(&self.attentions) {
            h = attn.forward(&h)?;
            h = resnet.forward(&h, image_only_indicator, num_frames)?;
        }
        Ok(h)
    }
}

/// Temporal Decoder for SVD VAE
#[derive(Debug)]
pub struct TemporalDecoder {
    conv_in: Conv2d,
    mid_block: MidBlockTemporalDecoder,
    up_blocks: Vec<UpBlockTemporalDecoder>,
    conv_norm_out: candle_nn::GroupNorm,
    conv_out: Conv2d,
    time_conv_out: Conv3d,
}

impl TemporalDecoder {
    pub fn new(vb: VarBuilder, config: &SvdVaeConfig) -> Result<Self> {
        let reversed_channels: Vec<_> = config.block_out_channels.iter().rev().copied().collect();

        let conv_in = conv2d(
            config.latent_channels,
            reversed_channels[0],
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_in"),
        )?;

        let mid_block = MidBlockTemporalDecoder::new(
            vb.pp("mid_block"),
            reversed_channels[0],
            config.layers_per_block,
        )?;

        let mut up_blocks = Vec::with_capacity(config.block_out_channels.len());
        let mut output_channel = reversed_channels[0];

        for (i, &out_ch) in reversed_channels.iter().enumerate() {
            let prev_output_channel = output_channel;
            output_channel = out_ch;
            let is_final = i == config.block_out_channels.len() - 1;

            up_blocks.push(UpBlockTemporalDecoder::new(
                vb.pp("up_blocks").pp(i),
                prev_output_channel,
                output_channel,
                config.layers_per_block + 1,
                !is_final,
            )?);
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

        let time_conv_out = Conv3d::new(
            vb.pp("time_conv_out"),
            config.out_channels,
            config.out_channels,
            (3, 1, 1),
            (1, 0, 0),
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            time_conv_out,
        })
    }

    pub fn forward(
        &self,
        z: &Tensor,
        image_only_indicator: &Tensor,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;

        h = self
            .mid_block
            .forward(&h, image_only_indicator, num_frames)?;

        for up_block in &self.up_blocks {
            h = up_block.forward(&h, image_only_indicator, num_frames)?;
        }

        h = self.conv_norm_out.forward(&h)?;
        h = candle_nn::ops::silu(&h)?;
        h = self.conv_out.forward(&h)?;

        // Apply temporal conv
        let (batch_frames, c, height, width) = h.dims4()?;
        let batch_size = batch_frames / num_frames;

        // Reshape to [B, C, T, H, W]
        let h = h
            .reshape((batch_size, num_frames, c, height, width))?
            .permute((0, 2, 1, 3, 4))?;

        let h = self.time_conv_out.forward(&h)?;

        // Reshape back to [B*T, C, H, W]
        let h = h
            .permute((0, 2, 1, 3, 4))?
            .reshape((batch_frames, c, height, width))?;

        Ok(h)
    }
}
