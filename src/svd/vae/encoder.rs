//! VAE Encoder for SVD
//!
//! Standard 2D encoder with downsampling blocks.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

use crate::svd::config::SvdVaeConfig;

/// ResnetBlock2D for encoder
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

        let residual = match &self.conv_shortcut {
            Some(conv) => conv.forward(residual)?,
            None => residual.clone(),
        };
        h + residual
    }
}

/// Downsample2D
#[derive(Debug)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let conv = conv2d(
            channels,
            channels,
            3,
            Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Attention block for encoder mid_block
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

/// DownBlock for encoder
#[derive(Debug)]
struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsamplers: Option<Downsample2D>,
}

impl DownEncoderBlock2D {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_downsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                vb.pp("resnets").pp(i),
                in_ch,
                out_channels,
            )?);
        }

        let downsamplers = if add_downsample {
            Some(Downsample2D::new(
                vb.pp("downsamplers").pp("0"),
                out_channels,
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            downsamplers,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(down) = &self.downsamplers {
            h = down.forward(&h)?;
        }
        Ok(h)
    }
}

/// MidBlock for encoder
#[derive(Debug)]
struct MidBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<AttentionBlock>,
}

impl MidBlock2D {
    fn new(vb: VarBuilder, channels: usize, num_layers: usize) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers + 1);
        let mut attentions = Vec::with_capacity(num_layers);

        resnets.push(ResnetBlock2D::new(
            vb.pp("resnets").pp("0"),
            channels,
            channels,
        )?);

        for i in 0..num_layers {
            attentions.push(AttentionBlock::new(vb.pp("attentions").pp(i), channels)?);
            resnets.push(ResnetBlock2D::new(
                vb.pp("resnets").pp(i + 1),
                channels,
                channels,
            )?);
        }

        Ok(Self {
            resnets,
            attentions,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.resnets[0].forward(x)?;
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            h = attn.forward(&h)?;
            h = resnet.forward(&h)?;
        }
        Ok(h)
    }
}

/// VAE Encoder
#[derive(Debug)]
pub struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: MidBlock2D,
    conv_norm_out: candle_nn::GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(vb: VarBuilder, config: &SvdVaeConfig) -> Result<Self> {
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

        let mut down_blocks = Vec::with_capacity(config.block_out_channels.len());
        let mut output_channel = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = out_ch;
            let is_final = i == config.block_out_channels.len() - 1;

            down_blocks.push(DownEncoderBlock2D::new(
                vb.pp("down_blocks").pp(i),
                input_channel,
                output_channel,
                config.layers_per_block,
                !is_final,
            )?);
        }

        let mid_block = MidBlock2D::new(
            vb.pp("mid_block"),
            *config.block_out_channels.last().unwrap(),
            1, // num_layers for attention
        )?;

        let conv_norm_out = candle_nn::group_norm(
            32,
            *config.block_out_channels.last().unwrap(),
            1e-6,
            vb.pp("conv_norm_out"),
        )?;

        let conv_out = conv2d(
            *config.block_out_channels.last().unwrap(),
            2 * config.latent_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        for down_block in &self.down_blocks {
            h = down_block.forward(&h)?;
        }

        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward(&h)?;
        h = candle_nn::ops::silu(&h)?;
        self.conv_out.forward(&h)
    }
}
