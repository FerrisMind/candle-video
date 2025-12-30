//! UNet Spatio-Temporal Blocks for SVD
//!
//! Down, Mid, and Up blocks with temporal processing.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

use super::resnet::SpatioTemporalResBlock;
use super::transformer::TransformerSpatioTemporalModel;

/// Downsample 2D with convolution
#[derive(Debug)]
pub struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Upsample 2D with convolution
#[derive(Debug)]
pub struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&x)
    }
}

/// Down block without attention
#[derive(Debug)]
pub struct DownBlockSpatioTemporal {
    resnets: Vec<SpatioTemporalResBlock>,
    downsamplers: Option<Downsample2D>,
}

impl DownBlockSpatioTemporal {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        temb_channels: Option<usize>,
        add_downsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(SpatioTemporalResBlock::new(
                vb.pp("resnets").pp(i),
                in_ch,
                out_channels,
                temb_channels,
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

    pub fn forward(
        &self,
        x: &Tensor,
        temb: Option<&Tensor>,
        image_only_indicator: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut output_states = Vec::new();
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h, temb, image_only_indicator, num_frames)?;
            output_states.push(h.clone());
        }

        if let Some(down) = &self.downsamplers {
            h = down.forward(&h)?;
            output_states.push(h.clone());
        }

        Ok((h, output_states))
    }
}

/// Down block with cross-attention
#[derive(Debug)]
pub struct CrossAttnDownBlockSpatioTemporal {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Vec<TransformerSpatioTemporalModel>,
    downsamplers: Option<Downsample2D>,
}

impl CrossAttnDownBlockSpatioTemporal {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        num_attention_heads: usize,
        cross_attention_dim: usize,
        temb_channels: Option<usize>,
        add_downsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(SpatioTemporalResBlock::new(
                vb.pp("resnets").pp(i),
                in_ch,
                out_channels,
                temb_channels,
            )?);
            attentions.push(TransformerSpatioTemporalModel::new(
                vb.pp("attentions").pp(i),
                out_channels,
                1,
                num_attention_heads,
                cross_attention_dim,
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
            attentions,
            downsamplers,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        image_only_indicator: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut output_states = Vec::new();
        let mut h = x.clone();

        for (resnet, attn) in self.resnets.iter().zip(&self.attentions) {
            h = resnet.forward(&h, temb, image_only_indicator, num_frames)?;
            h = attn.forward(&h, encoder_hidden_states, num_frames)?;
            output_states.push(h.clone());
        }

        if let Some(down) = &self.downsamplers {
            h = down.forward(&h)?;
            output_states.push(h.clone());
        }

        Ok((h, output_states))
    }
}

/// Mid block
#[derive(Debug)]
pub struct UNetMidBlockSpatioTemporal {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Vec<TransformerSpatioTemporalModel>,
}

impl UNetMidBlockSpatioTemporal {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        temb_channels: Option<usize>,
        num_attention_heads: usize,
        cross_attention_dim: usize,
    ) -> Result<Self> {
        let resnets = vec![
            SpatioTemporalResBlock::new(
                vb.pp("resnets").pp("0"),
                in_channels,
                in_channels,
                temb_channels,
            )?,
            SpatioTemporalResBlock::new(
                vb.pp("resnets").pp("1"),
                in_channels,
                in_channels,
                temb_channels,
            )?,
        ];
        let attentions = vec![TransformerSpatioTemporalModel::new(
            vb.pp("attentions").pp("0"),
            in_channels,
            1,
            num_attention_heads,
            cross_attention_dim,
        )?];
        Ok(Self {
            resnets,
            attentions,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        image_only_indicator: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut h = self.resnets[0].forward(x, temb, image_only_indicator, num_frames)?;
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            h = attn.forward(&h, encoder_hidden_states, num_frames)?;
            h = resnet.forward(&h, temb, image_only_indicator, num_frames)?;
        }
        Ok(h)
    }
}

/// Up block without attention - takes list of input channel sizes for each resnet
#[derive(Debug)]
pub struct UpBlockSpatioTemporal {
    resnets: Vec<SpatioTemporalResBlock>,
    upsamplers: Option<Upsample2D>,
}

impl UpBlockSpatioTemporal {
    pub fn new(
        vb: VarBuilder,
        in_channels_list: &[usize], // Input channels for each resnet
        out_channels: usize,
        temb_channels: Option<usize>,
        add_upsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(in_channels_list.len());
        for (i, &in_ch) in in_channels_list.iter().enumerate() {
            resnets.push(SpatioTemporalResBlock::new(
                vb.pp("resnets").pp(i),
                in_ch,
                out_channels,
                temb_channels,
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

    pub fn forward(
        &self,
        x: &Tensor,
        res_hidden_states: &mut Vec<Tensor>,
        temb: Option<&Tensor>,
        image_only_indicator: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, resnet) in self.resnets.iter().enumerate() {
            let res_state = res_hidden_states
                .pop()
                .expect("Not enough skip connections");
            
            // DEBUG: Check skip connection before cat
            if std::env::var("DEBUG_UNET").is_ok()
                && let Ok(f) = res_state.flatten_all().and_then(|f| f.to_dtype(candle_core::DType::F32))
                && let Ok(v) = f.to_vec1::<f32>()
                && v.iter().any(|x| x.is_nan() || x.is_infinite())
            {
                println!("      [UP_BLOCK] res_state[{}] has NaN/Inf!", i);
            }
            
            h = Tensor::cat(&[&h, &res_state], 1)?;
            h = resnet.forward(&h, temb, image_only_indicator, num_frames)?;
            
            // DEBUG: Check after resnet
            if std::env::var("DEBUG_UNET").is_ok()
                && let Ok(f) = h.flatten_all().and_then(|f| f.to_dtype(candle_core::DType::F32))
                && let Ok(v) = f.to_vec1::<f32>()
                && v.iter().any(|x| x.is_nan() || x.is_infinite())
            {
                println!("      [UP_BLOCK] after resnet[{}] has NaN/Inf!", i);
            }
        }
        if let Some(up) = &self.upsamplers {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// Up block with cross-attention - takes list of input channel sizes for each resnet
#[derive(Debug)]
pub struct CrossAttnUpBlockSpatioTemporal {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Vec<TransformerSpatioTemporalModel>,
    upsamplers: Option<Upsample2D>,
}

impl CrossAttnUpBlockSpatioTemporal {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        in_channels_list: &[usize], // Input channels for each resnet
        out_channels: usize,
        num_attention_heads: usize,
        cross_attention_dim: usize,
        temb_channels: Option<usize>,
        add_upsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(in_channels_list.len());
        let mut attentions = Vec::with_capacity(in_channels_list.len());

        for (i, &in_ch) in in_channels_list.iter().enumerate() {
            resnets.push(SpatioTemporalResBlock::new(
                vb.pp("resnets").pp(i),
                in_ch,
                out_channels,
                temb_channels,
            )?);
            attentions.push(TransformerSpatioTemporalModel::new(
                vb.pp("attentions").pp(i),
                out_channels,
                1,
                num_attention_heads,
                cross_attention_dim,
            )?);
        }

        let upsamplers = if add_upsample {
            Some(Upsample2D::new(vb.pp("upsamplers").pp("0"), out_channels)?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            upsamplers,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        res_hidden_states: &mut Vec<Tensor>,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        image_only_indicator: Option<&Tensor>,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for (resnet, attn) in self.resnets.iter().zip(&self.attentions) {
            let res_state = res_hidden_states
                .pop()
                .expect("Not enough skip connections");
            h = Tensor::cat(&[&h, &res_state], 1)?;
            h = resnet.forward(&h, temb, image_only_indicator, num_frames)?;
            h = attn.forward(&h, encoder_hidden_states, num_frames)?;
        }
        if let Some(up) = &self.upsamplers {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}
