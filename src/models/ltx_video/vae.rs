#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::interfaces::autoencoder::{AutoencoderError, VideoAutoencoder};
use crate::interfaces::autoencoder_mixin::AutoencoderMixin;
use crate::interfaces::video_types::VideoLatents;
use crate::models::ltx_video::t2v_pipeline::{VaeConfig, VaeLtxVideo};
use crate::ops::conv3d::{Conv3d, Conv3dConfig, PaddingMode};
use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, LayerNorm, LayerNormConfig, Linear, RmsNorm, VarBuilder, ops};

use serde::{Deserialize, Serialize};

pub const DEBUG_VAE: bool = false;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoencoderKLLtxVideoConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub decoder_block_out_channels: Vec<usize>,
    #[serde(alias = "spatio_temporal_scaling")]
    pub spatiotemporal_scaling: Vec<bool>,
    #[serde(alias = "decoder_spatio_temporal_scaling")]
    pub decoder_spatiotemporal_scaling: Vec<bool>,
    pub layers_per_block: Vec<usize>,
    pub decoder_layers_per_block: Vec<usize>,
    pub patch_size: usize,
    pub patch_size_t: usize,
    #[serde(alias = "resnet_norm_eps")]
    pub resnet_eps: f64,
    pub scaling_factor: f64,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
    pub decoder_inject_noise: Vec<bool>,
    #[serde(alias = "upsample_residual")]
    pub decoder_upsample_residual: Vec<bool>,
    #[serde(alias = "upsample_factor")]
    pub decoder_upsample_factor: Vec<usize>,
    pub timestep_conditioning: bool,
    #[serde(default)]
    pub latents_mean: Vec<f32>,
    #[serde(default)]
    pub latents_std: Vec<f32>,
    #[serde(alias = "downsample_type")]
    pub downsample_types: Vec<String>,
    #[serde(alias = "encoder_causal")]
    pub is_causal: bool,
    pub decoder_causal: bool,
}

impl Default for AutoencoderKLLtxVideoConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            block_out_channels: vec![128, 256, 512, 1024, 2048],
            decoder_block_out_channels: vec![256, 512, 1024],
            spatiotemporal_scaling: vec![true, true, true, true],
            decoder_spatiotemporal_scaling: vec![true, true, true],
            layers_per_block: vec![4, 6, 6, 2, 2],
            decoder_layers_per_block: vec![5, 5, 5, 5],
            patch_size: 4,
            patch_size_t: 1,
            resnet_eps: 1e-6,
            scaling_factor: 1.0,
            spatial_compression_ratio: 32,
            temporal_compression_ratio: 8,
            decoder_inject_noise: vec![false, false, false, false],
            decoder_upsample_residual: vec![true, true, true],
            decoder_upsample_factor: vec![2, 2, 2],
            timestep_conditioning: true,
            latents_mean: vec![0.0; 128],
            latents_std: vec![1.0; 128],
            downsample_types: vec![
                "spatial".into(),
                "temporal".into(),
                "spatiotemporal".into(),
                "spatiotemporal".into(),
            ],
            is_causal: true,
            decoder_causal: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecoderOutput {
    pub sample: Tensor,
}

#[derive(Clone, Debug)]
pub struct AutoencoderKLOutput {
    pub latent_dist: DiagonalGaussianDistribution,
}

#[derive(Clone, Debug)]
pub struct DiagonalGaussianDistribution {
    pub mean: Tensor,
    pub logvar: Tensor,
}

impl DiagonalGaussianDistribution {
    pub fn new(moments: &Tensor) -> Result<Self> {
        let (_b, ch2, _t, _h, _w) = moments.dims5()?;
        if ch2 % 2 != 0 {
            candle_core::bail!("moments channels must be even, got {}", ch2)
        }
        let ch = ch2 / 2;
        let mean = moments.i((.., 0..ch, .., .., ..))?;
        let logvar = moments.i((.., ch..(2 * ch), .., .., ..))?;
        Ok(Self { mean, logvar })
    }

    pub fn mode(&self) -> Result<Tensor> {
        Ok(self.mean.clone())
    }

    pub fn sample(&self) -> Result<Tensor> {
        let eps = Tensor::randn(0f32, 1f32, self.mean.shape(), self.mean.device())?
            .to_dtype(self.mean.dtype())?;
        let std = (self.logvar.affine(0.5, 0.)?).exp()?;
        self.mean.add(&std.mul(&eps)?)
    }
}

fn rmsnorm_channels_first(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    x.permute((0, 2, 3, 4, 1))?
        .apply(norm)?
        .permute((0, 4, 1, 2, 3))
}

fn layernorm_channels_first(norm: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    x.permute((0, 2, 3, 4, 1))?
        .apply(norm)?
        .permute((0, 4, 1, 2, 3))
}

fn silu(x: &Tensor) -> Result<Tensor> {
    ops::silu(x)
}

fn cat_dim(xs: &[Tensor], dim: usize) -> Result<Tensor> {
    let refs: Vec<&Tensor> = xs.iter().collect();
    Tensor::cat(&refs, dim)
}

fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let device = timesteps.device();
    let dtype = timesteps.dtype();

    let max_period = 10000f64;
    let downscale_freq_shift = 0.0;

    let exponent_coef = -(max_period.ln()) / (half_dim as f64 - downscale_freq_shift);
    let emb = (Tensor::arange(0u32, half_dim as u32, device)?
        .to_dtype(DType::F32)?
        .affine(exponent_coef, 0.0))?
    .exp()?;

    let timesteps_f = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;
    let emb = timesteps_f.broadcast_mul(&emb.unsqueeze(0)?)?;

    let sin_emb = emb.sin()?;
    let cos_emb = emb.cos()?;

    Tensor::cat(&[&cos_emb, &sin_emb], 1)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = candle_nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        if DEBUG_VAE && let Ok(w) = self.linear_1.weight().flatten_all()?.to_vec1::<f32>() {
            println!(
                "[DEBUG] TimestepEmbedder linear_1 weight first 5: {:?}",
                &w[..5.min(w.len())]
            );
        }
        let h = self.linear_1.forward(t)?;
        if DEBUG_VAE && let Ok(vals) = h.flatten_all()?.to_vec1::<f32>() {
            println!(
                "[DEBUG] TimestepEmbedder after linear_1 first 5: {:?}",
                &vals[..5.min(vals.len())]
            );
        }
        let h = silu(&h)?;
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
pub struct CombinedTimestepEmbedder {
    timestep_embedder: TimestepEmbedder,
}

impl CombinedTimestepEmbedder {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder =
            TimestepEmbedder::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self { timestep_embedder })
    }

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: DType) -> Result<Tensor> {
        let timesteps_proj = get_timestep_embedding(timestep, 256)?;

        if DEBUG_VAE && let Ok(vals) = timesteps_proj.flatten_all()?.to_vec1::<f32>() {
            println!(
                "[DEBUG] Rust sinusoidal embedding first 10: {:?}",
                &vals[..10.min(vals.len())]
            );
            println!(
                "[DEBUG] Rust sinusoidal embedding mean: {:.6}",
                vals.iter().sum::<f32>() / vals.len() as f32
            );
        }

        self.timestep_embedder
            .forward(&timesteps_proj.to_dtype(hidden_dtype)?)
    }
}

pub type LtxVideoCausalConv3d = Conv3d;

pub fn create_ltx_causal_conv3d(
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize, usize),
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    is_causal: bool,
    vb: VarBuilder,
) -> Result<Conv3d> {
    let (kt, kh, kw) = kernel;
    let (_st, _sh, _sw) = stride;
    let (_dt, _dh, _dw) = dilation;

    let ph = kh / 2;
    let pw = kw / 2;

    let config = if is_causal {
        Conv3dConfig::causal(kernel)
            .with_stride(stride)
            .with_padding((0, ph, pw))
            .with_dilation(dilation)
            .with_groups(groups)
    } else {
        let pt = (kt - 1) / 2;
        Conv3dConfig::new(kernel)
            .with_stride(stride)
            .with_padding((pt, ph, pw))
            .with_dilation(dilation)
            .with_groups(groups)
            .with_padding_mode(PaddingMode::Replicate)
    };

    Conv3d::new_with_conv_submodule(in_channels, out_channels, config, vb)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DownsampleType {
    Conv,
    Spatial,
    Temporal,
    Spatiotemporal,
}

impl DownsampleType {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "spatial" => Self::Spatial,
            "temporal" => Self::Temporal,
            "spatiotemporal" => Self::Spatiotemporal,
            _ => Self::Conv,
        }
    }

    pub fn stride(&self) -> (usize, usize, usize) {
        match self {
            Self::Conv => (2, 2, 2),
            Self::Spatial => (1, 2, 2),
            Self::Temporal => (2, 1, 1),
            Self::Spatiotemporal => (2, 2, 2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoDownsampler3d {
    stride: (usize, usize, usize),
    group_size: usize,
    conv: Conv3d,
}

impl LtxVideoDownsampler3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        is_causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (st, sh, sw) = stride;
        let group_size = (in_channels * st * sh * sw) / out_channels;
        let conv_out_channels = out_channels / (st * sh * sw);

        let conv = create_ltx_causal_conv3d(
            in_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv"),
        )?;

        Ok(Self {
            stride,
            group_size,
            conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (st, sh, sw) = self.stride;
        let (b, c, _t, _h, _w) = x.dims5()?;

        let padded = if st > 1 {
            let pad_slice = x.i((.., .., ..(st - 1), .., ..))?;
            Tensor::cat(&[&pad_slice, x], 2)?
        } else {
            x.clone()
        };
        let (_, _, t_pad, h_pad, w_pad) = padded.dims5()?;

        let t_new = t_pad / st;
        let h_new = h_pad / sh;
        let w_new = w_pad / sw;

        let residual = padded
            .reshape(&[b, c, t_new, st, h_new, sh, w_new, sw])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?
            .reshape((b, c * st * sh * sw, t_new, h_new, w_new))?;

        let residual = residual
            .reshape(&[
                b,
                c * st * sh * sw / self.group_size,
                self.group_size,
                t_new,
                h_new,
                w_new,
            ])?
            .mean(2)?;

        let conv_out = self.conv.forward(&padded)?;
        let (_, c_conv, _, _, _) = conv_out.dims5()?;

        let hidden = conv_out
            .reshape(&[b, c_conv, t_new, st, h_new, sh, w_new, sw])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?
            .reshape((b, c_conv * st * sh * sw, t_new, h_new, w_new))?;

        hidden.add(&residual)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoResnetBlock3d {
    norm1: Option<RmsNorm>,
    conv1: Conv3d,
    norm2: Option<RmsNorm>,
    _dropout: f64,
    conv2: Conv3d,

    norm3: Option<LayerNorm>,
    conv_shortcut: Option<Conv3d>,

    per_channel_scale1: Option<Tensor>,
    per_channel_scale2: Option<Tensor>,

    scale_shift_table: Option<Tensor>,
}

impl LtxVideoResnetBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        dropout: f64,
        eps: f64,
        elementwise_affine: bool,
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let load_norm = |name: &str, size: usize| -> Result<RmsNorm> {
            if elementwise_affine {
                let norm_res = candle_nn::rms_norm(size, 1e-8, vb.pp(name));
                if let Ok(norm) = norm_res {
                    return Ok(norm);
                }
            }

            let ones = Tensor::ones((size,), vb.dtype(), vb.device())?;
            Ok(RmsNorm::new(ones, 1e-8))
        };

        let norm1 = Some(load_norm("norm1", in_channels)?);
        let conv1 = create_ltx_causal_conv3d(
            in_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv1"),
        )?;

        let norm2 = Some(load_norm("norm2", out_channels)?);
        let conv2 = create_ltx_causal_conv3d(
            out_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv2"),
        )?;

        let (norm3, conv_shortcut) = if in_channels != out_channels {
            let lncfg = LayerNormConfig {
                eps,
                affine: elementwise_affine,
                ..Default::default()
            };
            let norm3 = candle_nn::layer_norm(in_channels, lncfg, vb.pp("norm3")).ok();
            let conv_shortcut = create_ltx_causal_conv3d(
                in_channels,
                out_channels,
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                1,
                is_causal,
                vb.pp("conv_shortcut"),
            )?;
            (norm3, Some(conv_shortcut))
        } else {
            (None, None)
        };

        let per_channel_scale1 = if inject_noise {
            vb.pp("per_channel_scale1")
                .get((in_channels, 1, 1), "weight")
                .ok()
        } else {
            None
        };
        let per_channel_scale2 = if inject_noise {
            vb.pp("per_channel_scale2")
                .get((in_channels, 1, 1), "weight")
                .ok()
        } else {
            None
        };

        let scale_shift_table = if timestep_conditioning {
            vb.get((4, in_channels), "scale_shift_table").ok()
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            _dropout: dropout,
            conv2,
            norm3,
            conv_shortcut,
            per_channel_scale1,
            per_channel_scale2,
            scale_shift_table,
        })
    }

    fn maybe_apply_scale_shift(
        &self,
        x: Tensor,
        temb: Option<&Tensor>,
        stage: usize,
    ) -> Result<Tensor> {
        let Some(tbl) = &self.scale_shift_table else {
            return Ok(x);
        };
        let Some(temb) = temb else {
            return Ok(x);
        };

        let (b, temb_dim, _, _, _) = temb.dims5()?;
        let c = tbl.dims2()?.1;
        if temb_dim != 4 * c {
            candle_core::bail!("temb dim mismatch: got {}, expected {}", temb_dim, 4 * c)
        }
        let temb = temb
            .reshape((b, 4, c, 1, 1, 1))?
            .broadcast_add(&tbl.unsqueeze(0)?.unsqueeze(3)?.unsqueeze(4)?.unsqueeze(5)?)?;

        let shift = temb.i((.., stage * 2, .., .., .., ..))?;
        let scale = temb.i((.., stage * 2 + 1, .., .., .., ..))?;

        x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)
    }

    fn maybe_inject_noise(&self, x: Tensor, pcs: &Option<Tensor>) -> Result<Tensor> {
        let Some(scale) = pcs else {
            return Ok(x);
        };

        let (_b, _c, _t, h, w) = x.dims5()?;
        let noise = Tensor::randn(0f32, 1f32, (h, w), x.device())?.to_dtype(x.dtype())?;

        let noise = noise.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?;

        let scale = scale.unsqueeze(0)?.unsqueeze(2)?;
        x.add(&(noise.broadcast_mul(&scale)?))
    }

    pub fn forward(&self, inputs: &Tensor, temb: Option<&Tensor>, _train: bool) -> Result<Tensor> {
        let mut h = inputs.clone();

        if let Some(ref norm1) = self.norm1 {
            h = rmsnorm_channels_first(norm1, &h)?;
        }

        let has_sst = self.scale_shift_table.is_some();
        let has_temb = temb.is_some();

        h = self.maybe_apply_scale_shift(h, temb, 0)?;
        if DEBUG_VAE
            && has_sst
            && let Ok(vals) = h.flatten_all()?.to_vec1::<f32>()
        {
            println!(
                "[DEBUG] Resnet h after scale0: sst={}, temb={} mean: {:.4}",
                has_sst,
                has_temb,
                vals.iter().sum::<f32>() / vals.len() as f32
            );
        }

        h = silu(&h)?;

        h = self.conv1.forward(&h)?;

        h = self.maybe_inject_noise(h, &self.per_channel_scale1)?;

        if let Some(ref norm2) = self.norm2 {
            h = rmsnorm_channels_first(norm2, &h)?;
        }
        h = self.maybe_apply_scale_shift(h, temb, 1)?;
        if DEBUG_VAE
            && has_sst
            && let Ok(vals) = h.flatten_all()?.to_vec1::<f32>()
        {
            println!(
                "[DEBUG] Resnet h after scale1 mean: {:.4}",
                vals.iter().sum::<f32>() / vals.len() as f32
            );
        }
        h = silu(&h)?;

        h = self.conv2.forward(&h)?;

        h = self.maybe_inject_noise(h, &self.per_channel_scale2)?;

        let mut x = inputs.clone();
        if let Some(n3) = &self.norm3 {
            x = layernorm_channels_first(n3, &x)?;
        }
        if let Some(cs) = &self.conv_shortcut {
            x = cs.forward(&x)?;
        }
        let result = h.add(&x)?;

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub enum Downsampler {
    Conv(Conv3d),
    PixelUnshuffle(LtxVideoDownsampler3d),
}

impl Downsampler {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Downsampler::Conv(c) => c.forward(x),
            Downsampler::PixelUnshuffle(p) => p.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoDownBlock3d {
    resnets: Vec<LtxVideoResnetBlock3d>,
    downsamplers: Option<Vec<Downsampler>>,
    conv_out: Option<LtxVideoResnetBlock3d>,
    pub gradient_checkpointing: bool,
}

impl LtxVideoDownBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        spatiotemporal_scale: bool,
        is_causal: bool,
        downsample_type: DownsampleType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(LtxVideoResnetBlock3d::new(
                in_channels,
                in_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                false,
                false,
                vb.pp(format!("resnets.{i}")),
            )?);
        }

        let downsamplers = if spatiotemporal_scale {
            let ds = match downsample_type {
                DownsampleType::Conv => Downsampler::Conv(create_ltx_causal_conv3d(
                    in_channels,
                    in_channels,
                    (3, 3, 3),
                    (2, 2, 2),
                    (1, 1, 1),
                    1,
                    is_causal,
                    vb.pp("downsamplers.0").pp("conv"),
                )?),
                _ => {
                    let stride = downsample_type.stride();
                    Downsampler::PixelUnshuffle(LtxVideoDownsampler3d::new(
                        in_channels,
                        out_channels,
                        stride,
                        is_causal,
                        vb.pp("downsamplers.0"),
                    )?)
                }
            };
            Some(vec![ds])
        } else {
            None
        };

        let conv_out = if in_channels != out_channels && downsample_type == DownsampleType::Conv {
            LtxVideoResnetBlock3d::new(
                in_channels,
                out_channels,
                dropout,
                resnet_eps,
                true,
                is_causal,
                false,
                false,
                vb.pp("conv_out"),
            )
            .ok()
        } else {
            None
        };

        Ok(Self {
            resnets,
            downsamplers,
            conv_out,
            gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();
        for r in self.resnets.iter() {
            h = r.forward(&h, temb, train)?;
        }
        if let Some(ds) = &self.downsamplers {
            for d in ds.iter() {
                h = d.forward(&h)?;
            }
        }
        if let Some(co) = &self.conv_out {
            h = co.forward(&h, temb, train)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoMidBlock3d {
    resnets: Vec<LtxVideoResnetBlock3d>,
    pub gradient_checkpointing: bool,
    time_embedder: Option<CombinedTimestepEmbedder>,
}

impl LtxVideoMidBlock3d {
    pub fn new(
        in_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(LtxVideoResnetBlock3d::new(
                in_channels,
                in_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                inject_noise,
                timestep_conditioning,
                vb.pp(format!("resnets.{i}")),
            )?);
        }

        let time_embedder = if timestep_conditioning {
            let emb_dim = in_channels * 4;
            CombinedTimestepEmbedder::new(emb_dim, vb.pp("time_embedder")).ok()
        } else {
            None
        };

        Ok(Self {
            resnets,
            time_embedder,
            gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();

        let temb_proj = if let (Some(te), Some(t)) = (&self.time_embedder, temb) {
            if DEBUG_VAE && let Ok(vals) = t.flatten_all()?.to_vec1::<f32>() {
                println!("[DEBUG] mid_block time_embedder input (temb): {:?}", vals);
            }

            let emb = te.forward(t, h.dtype())?;
            if DEBUG_VAE && let Ok(vals) = emb.flatten_all()?.to_vec1::<f32>() {
                println!(
                    "[DEBUG] mid_block time_embedder output shape: {:?}, first 10: {:?}, mean: {:.4}",
                    emb.dims(),
                    &vals[..10.min(vals.len())],
                    vals.iter().sum::<f32>() / vals.len() as f32
                );
            }
            let batch_size = h.dims5()?.0;
            let emb_dim = emb.dims2()?.1;
            Some(emb.reshape((batch_size, emb_dim, 1, 1, 1))?)
        } else {
            if DEBUG_VAE {
                println!(
                    "[DEBUG] mid_block time_embedder: SKIPPED (te={}, temb={})",
                    self.time_embedder.is_some(),
                    temb.is_some()
                );
            }
            None
        };

        for r in self.resnets.iter() {
            h = r.forward(&h, temb_proj.as_ref(), train)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoUpsampler3d {
    stride_t: usize,
    stride_h: usize,
    stride_w: usize,
    residual: bool,

    channel_repeats: usize,
    conv: Conv3d,
}

impl LtxVideoUpsampler3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        is_causal: bool,
        residual: bool,
        _upscale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (st, sh, sw) = stride;
        let stride_product = st * sh * sw;

        let conv_out_channels = out_channels * stride_product;

        let channel_repeats = conv_out_channels / in_channels;

        let conv = create_ltx_causal_conv3d(
            in_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv"),
        )?;
        Ok(Self {
            stride_t: st,
            stride_h: sh,
            stride_w: sw,
            residual,
            channel_repeats,
            conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = x.dims5()?;
        let st = self.stride_t;
        let sh = self.stride_h;
        let sw = self.stride_w;

        if DEBUG_VAE {
            println!("[DEBUG] upsampler input: {:?}", x.dims());
        }

        let residual = if self.residual {
            let cprime = x.dims5()?.1;
            let c_out = cprime / (st * sh * sw);

            let x2 = x.reshape(&[b, c_out, st, sh, sw, t, h, w])?;

            let x2 = x2.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?.contiguous()?;

            let x2 = x2.reshape(&[b, c_out, t, st, h, sh, w * sw])?;

            let x2 = x2.reshape(&[b, c_out, t, st, h * sh, w * sw])?;

            let x2 = x2.reshape(&[b, c_out, t * st, h * sh, w * sw])?;

            let x2 = if self.channel_repeats > 1 {
                x2.repeat((1, self.channel_repeats, 1, 1, 1))?
            } else {
                x2
            };

            let x2 = x2.i((.., .., (st - 1).., .., ..))?;
            Some(x2)
        } else {
            None
        };

        let h0 = self.conv.forward(x)?;
        let h0 = h0.contiguous()?;
        if DEBUG_VAE {
            println!("[DEBUG] after conv: {:?}", h0.dims());
        }

        let (_b2, c2, t2, h2, w2) = h0.dims5()?;
        let c_out = c2 / (st * sh * sw);
        if DEBUG_VAE {
            println!("[DEBUG] c_out (C' after shuffle): {}", c_out);
        }

        let h1 = h0.reshape(&[b, c_out, st, sh, sw, t2, h2, w2])?;
        if DEBUG_VAE {
            println!("[DEBUG] after reshape: {:?}", h1.dims());
        }

        let h1 = h1.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?.contiguous()?;
        if DEBUG_VAE {
            println!("[DEBUG] after permute: {:?}", h1.dims());
        }

        let h1 = h1.reshape(&[b, c_out, t2, st, h2, sh, w2 * sw])?;

        let h1 = h1.reshape(&[b, c_out, t2, st, h2 * sh, w2 * sw])?;

        let h1 = h1.reshape(&[b, c_out, t2 * st, h2 * sh, w2 * sw])?;

        let h1 = h1.i((.., .., (st - 1).., .., ..))?;

        let h1 = if let Some(r) = residual {
            h1.add(&r)?
        } else {
            h1
        };
        Ok(h1)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoUpBlock3d {
    conv_in: Option<LtxVideoResnetBlock3d>,
    upsamplers: Option<Vec<LtxVideoUpsampler3d>>,
    resnets: Vec<LtxVideoResnetBlock3d>,
    time_embedder: Option<CombinedTimestepEmbedder>,
    pub gradient_checkpointing: bool,
}

impl LtxVideoUpBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        spatiotemporal_scale: bool,
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        upsampler_residual: bool,
        up_scale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_in = if in_channels != out_channels {
            Some(LtxVideoResnetBlock3d::new(
                in_channels,
                out_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                inject_noise,
                timestep_conditioning,
                vb.pp("conv_in"),
            )?)
        } else {
            None
        };

        let upsamplers = if spatiotemporal_scale {
            Some(vec![LtxVideoUpsampler3d::new(
                out_channels * up_scale_factor,
                out_channels,
                (2, 2, 2),
                is_causal,
                upsampler_residual,
                up_scale_factor,
                vb.pp("upsamplers.0"),
            )?])
        } else {
            Some(vec![LtxVideoUpsampler3d::new(
                out_channels * up_scale_factor,
                out_channels,
                (1, 2, 2),
                is_causal,
                upsampler_residual,
                up_scale_factor,
                vb.pp("upsamplers.0"),
            )?])
        };

        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_c = out_channels;
            resnets.push(LtxVideoResnetBlock3d::new(
                in_c,
                out_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                inject_noise,
                timestep_conditioning,
                vb.pp(format!("resnets.{i}")),
            )?);
        }

        let time_embedder = if timestep_conditioning {
            let emb_dim = out_channels * 4;
            CombinedTimestepEmbedder::new(emb_dim, vb.pp("time_embedder")).ok()
        } else {
            None
        };

        Ok(Self {
            conv_in,
            upsamplers,
            resnets,
            time_embedder,
            gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();

        if let Some(ci) = &self.conv_in {
            h = ci.forward(&h, None, train)?;
        }

        let temb_proj = if let (Some(te), Some(t)) = (&self.time_embedder, temb) {
            let emb = te.forward(t, h.dtype())?;
            let batch_size = h.dims5()?.0;
            let emb_dim = emb.dims2()?.1;
            Some(emb.reshape((batch_size, emb_dim, 1, 1, 1))?)
        } else {
            None
        };

        if let Some(us) = &self.upsamplers {
            for u in us.iter() {
                h = u.forward(&h)?;
            }
        }

        for r in self.resnets.iter() {
            h = r.forward(&h, temb_proj.as_ref(), train)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoEncoder3d {
    patch_size: usize,
    patch_size_t: usize,
    conv_in: Conv3d,
    down_blocks: Vec<LtxVideoDownBlock3d>,
    mid_block: LtxVideoMidBlock3d,
    norm_out: Option<RmsNorm>,
    conv_act: Activation,
    conv_out: Conv3d,
    pub gradient_checkpointing: bool,
}

impl LtxVideoEncoder3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        block_out_channels: &[usize],
        spatiotemporal_scaling: &[bool],
        layers_per_block: &[usize],
        downsample_types: &[DownsampleType],
        patch_size: usize,
        patch_size_t: usize,
        resnet_eps: f64,
        is_causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channels_patched = in_channels * patch_size * patch_size * patch_size_t;
        let conv_in = create_ltx_causal_conv3d(
            in_channels_patched,
            block_out_channels[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::new();
        let n = block_out_channels.len() - 1;
        let mut current = block_out_channels[0];

        for i in 0..n {
            let outc = block_out_channels[i + 1];

            let ds_type = downsample_types
                .get(i)
                .copied()
                .unwrap_or(DownsampleType::Conv);

            let db = LtxVideoDownBlock3d::new(
                current,
                outc,
                layers_per_block[i],
                0.0,
                resnet_eps,
                spatiotemporal_scaling[i],
                is_causal,
                ds_type,
                vb.pp(format!("down_blocks.{i}")),
            )?;
            down_blocks.push(db);
            current = outc;
        }

        let mid_layers = *layers_per_block.last().unwrap_or(&1);
        let mid_block = LtxVideoMidBlock3d::new(
            current,
            mid_layers.saturating_sub(1),
            0.0,
            resnet_eps,
            is_causal,
            false,
            false,
            vb.pp("mid_block"),
        )?;

        let norm_out = if let Ok(norm) = candle_nn::rms_norm(current, 1e-8, vb.pp("norm_out")) {
            Some(norm)
        } else {
            let ones = Tensor::ones((current,), vb.dtype(), vb.device())?;
            Some(RmsNorm::new(ones, 1e-8))
        };
        let conv_act = Activation::Silu;
        let conv_out = create_ltx_causal_conv3d(
            current,
            out_channels + 1,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            patch_size,
            patch_size_t,
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_act,
            conv_out,
            gradient_checkpointing: false,
        })
    }

    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.patch_size;
        let pt = self.patch_size_t;
        let (b, c, f, h, w) = x.dims5()?;
        if f % pt != 0 || h % p != 0 || w % p != 0 {
            candle_core::bail!("input not divisible by patch sizes")
        }
        let post_f = f / pt;
        let post_h = h / p;
        let post_w = w / p;

        let x = x.reshape(&[b, c, post_f, pt, post_h, p, post_w, p])?;
        let x = x
            .permute(vec![0, 1, 3, 7, 5, 2, 4, 6])?
            .contiguous()?
            .reshape((b, c * pt * p * p, post_f, post_h, post_w))?;
        Ok(x)
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut h = self.patchify(x)?;
        h = self.conv_in.forward(&h)?;
        for db in self.down_blocks.iter() {
            h = db.forward(&h, None, train)?;
        }
        h = self.mid_block.forward(&h, None, train)?;

        if let Some(ref norm) = self.norm_out {
            h = rmsnorm_channels_first(norm, &h)?;
        }

        h = h.apply(&self.conv_act)?;
        h = self.conv_out.forward(&h)?;

        let (_b, ch, _t, _h, _w) = h.dims5()?;
        let last = h.i((.., (ch - 1), .., .., ..))?.unsqueeze(1)?;
        let rep = last.repeat((1, ch.saturating_sub(2), 1, 1, 1))?;
        cat_dim(&[h, rep], 1)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoDecoder3d {
    patch_size: usize,
    patch_size_t: usize,
    pub conv_in: Conv3d,
    pub mid_block: LtxVideoMidBlock3d,
    pub up_blocks: Vec<LtxVideoUpBlock3d>,
    pub norm_out: Option<RmsNorm>,
    pub conv_act: Activation,
    pub conv_out: Conv3d,

    pub time_embedder: Option<CombinedTimestepEmbedder>,
    pub scale_shift_table: Option<Tensor>,
    pub timestep_scale_multiplier: Option<Tensor>,
    pub gradient_checkpointing: bool,
}

impl LtxVideoDecoder3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        block_out_channels: &[usize],
        spatiotemporal_scaling: &[bool],
        layers_per_block: &[usize],
        patch_size: usize,
        patch_size_t: usize,
        resnet_eps: f64,
        is_causal: bool,
        inject_noise: &[bool],
        timestep_conditioning: bool,
        upsampler_residual: &[bool],
        upsample_factor: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut boc = block_out_channels.to_vec();
        boc.reverse();
        let mut sts = spatiotemporal_scaling.to_vec();
        sts.reverse();
        let mut lpb = layers_per_block.to_vec();
        lpb.reverse();

        let mut inj = inject_noise.to_vec();
        inj.reverse();
        let mut upr = upsampler_residual.to_vec();
        upr.reverse();
        let mut upf = upsample_factor.to_vec();
        upf.reverse();

        let conv_in = create_ltx_causal_conv3d(
            in_channels,
            boc[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_in"),
        )?;

        let mid_block = LtxVideoMidBlock3d::new(
            boc[0],
            lpb[0],
            0.0,
            resnet_eps,
            is_causal,
            inj[0],
            timestep_conditioning,
            vb.pp("mid_block"),
        )?;

        let mut up_blocks = Vec::new();
        let n = boc.len();
        let mut current_channels = 1024;

        for i in 0..n {
            let output_channel = boc[i] / upf[i];
            let input_channel = output_channel;

            let ub = LtxVideoUpBlock3d::new(
                input_channel,
                output_channel,
                lpb[i + 1],
                0.0,
                resnet_eps,
                sts[i],
                is_causal,
                inj[i + 1],
                timestep_conditioning,
                upr[i],
                upf[i],
                vb.pp(format!("up_blocks.{i}")),
            )?;
            up_blocks.push(ub);
            current_channels = output_channel;
        }

        let norm_out =
            if let Ok(norm) = candle_nn::rms_norm(current_channels, 1e-8, vb.pp("norm_out")) {
                Some(norm)
            } else {
                let ones = Tensor::ones((current_channels,), vb.dtype(), vb.device())?;
                Some(RmsNorm::new(ones, 1e-8))
            };
        let conv_act = Activation::Silu;

        let conv_out_channels = out_channels * patch_size * patch_size;
        let conv_out = create_ltx_causal_conv3d(
            current_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_out"),
        )?;

        let (time_embedder, scale_shift_table, timestep_scale_multiplier) = if timestep_conditioning
        {
            let emb_dim = current_channels * 2;
            let te = CombinedTimestepEmbedder::new(emb_dim, vb.pp("time_embedder")).ok();

            let sst = vb.get((2, current_channels), "scale_shift_table").ok();
            let tsm = vb.get((), "timestep_scale_multiplier").ok();
            (te, sst, tsm)
        } else {
            (None, None, None)
        };

        Ok(Self {
            patch_size,
            patch_size_t,
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_act,
            conv_out,
            time_embedder,
            scale_shift_table,
            timestep_scale_multiplier,
            gradient_checkpointing: false,
        })
    }

    pub fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = x.dims5()?;
        let p = self.patch_size;
        let pt = self.patch_size_t;
        let out_c = c / (pt * p * p);
        let x = x.reshape(&[b, out_c, pt, p, p, f, h, w])?;

        let x = x.permute(vec![0, 1, 5, 2, 6, 4, 7, 3])?;
        let x = x.contiguous()?;

        let x = x.reshape(&[b, out_c, f, pt, h, p, w * p])?;

        let x = x.reshape(&[b, out_c, f, pt, h * p, w * p])?;

        let x = x.reshape(&[b, out_c, f * pt, h * p, w * p])?;

        Ok(x)
    }

    pub fn forward(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let model_dtype = self.conv_in.weight().dtype();
        let z = z.to_dtype(model_dtype)?;
        let temb = match temb {
            Some(t) => Some(t.to_dtype(model_dtype)?),
            None => None,
        };

        let mut h = self.conv_in.forward(&z)?;

        let temb_scaled =
            if let (Some(tsm), Some(t)) = (&self.timestep_scale_multiplier, temb.as_ref()) {
                let t_flat = t.flatten_all()?;
                Some(t_flat.broadcast_mul(tsm)?)
            } else if let Some(t) = temb.as_ref() {
                Some(t.flatten_all()?)
            } else {
                None
            };
        let temb_for_blocks_ref = temb_scaled.as_ref();

        h = self.mid_block.forward(&h, temb_for_blocks_ref, train)?;

        for ub in self.up_blocks.iter() {
            h = ub.forward(&h, temb_for_blocks_ref, train)?;
        }

        if let Some(ref norm) = self.norm_out {
            h = rmsnorm_channels_first(norm, &h)?;
        }

        if let (Some(te), Some(sst), Some(temb_s)) =
            (&self.time_embedder, &self.scale_shift_table, &temb_scaled)
        {
            let temb_proj = te.forward(temb_s, h.dtype())?;

            let batch_size = h.dims5()?.0;
            let c = sst.dims2()?.1;
            let temb_shaped = temb_proj
                .reshape((batch_size, 2, c))?
                .broadcast_add(&sst.unsqueeze(0)?)?
                .unsqueeze(3)?
                .unsqueeze(4)?
                .unsqueeze(5)?;

            let shift = temb_shaped.i((.., 0, .., .., .., ..))?.squeeze(1)?;
            let scale = temb_shaped.i((.., 1, .., .., .., ..))?.squeeze(1)?;

            let h_shape = h.shape();
            let scale_b = scale.broadcast_as(h_shape)?;
            let shift_b = shift.broadcast_as(h_shape)?;

            h = h
                .broadcast_mul(&scale_b.affine(1.0, 1.0)?)?
                .broadcast_add(&shift_b)?;
        }

        h = h.apply(&self.conv_act)?;
        h = self.conv_out.forward(&h)?;
        self.unpatchify(&h)
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKLLtxVideo {
    pub encoder: LtxVideoEncoder3d,
    pub decoder: LtxVideoDecoder3d,
    pub quant_conv: Option<Conv3d>,
    pub post_quant_conv: Option<Conv3d>,

    pub latents_mean: Tensor,
    pub latents_std: Tensor,

    pub scaling_factor: f64,

    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,

    pub use_slicing: bool,
    pub use_tiling: bool,
    pub use_framewise_encoding: bool,
    pub use_framewise_decoding: bool,

    pub num_sample_frames_batch_size: usize,
    pub num_latent_frames_batch_size: usize,

    pub tile_sample_min_height: usize,
    pub tile_sample_min_width: usize,
    pub tile_sample_min_num_frames: usize,

    pub tile_sample_stride_height: usize,
    pub tile_sample_stride_width: usize,
    pub tile_sample_stride_num_frames: usize,

    pub config: AutoencoderKLLtxVideoConfig,
    pub vae_config: VaeConfig,
}

impl AutoencoderKLLtxVideo {
    pub fn new(config: AutoencoderKLLtxVideoConfig, vb: VarBuilder) -> Result<Self> {
        let ds_types: Vec<DownsampleType> = config
            .downsample_types
            .iter()
            .map(|s| DownsampleType::parse(s))
            .collect();

        let encoder = LtxVideoEncoder3d::new(
            config.in_channels,
            config.latent_channels,
            &config.block_out_channels,
            &config.spatiotemporal_scaling,
            &config.layers_per_block,
            &ds_types,
            config.patch_size,
            config.patch_size_t,
            config.resnet_eps,
            config.is_causal,
            vb.pp("encoder"),
        )?;

        let quant_conv = create_ltx_causal_conv3d(
            config.latent_channels * 2,
            config.latent_channels * 2,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            config.is_causal,
            vb.pp("quant_conv"),
        )
        .ok();

        let post_quant_conv = create_ltx_causal_conv3d(
            config.latent_channels,
            config.latent_channels,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            config.is_causal,
            vb.pp("post_quant_conv"),
        )
        .ok();

        let decoder = LtxVideoDecoder3d::new(
            config.latent_channels,
            config.out_channels,
            &config.decoder_block_out_channels,
            &config.decoder_spatiotemporal_scaling,
            &config.decoder_layers_per_block,
            config.patch_size,
            config.patch_size_t,
            config.resnet_eps,
            config.decoder_causal,
            &config.decoder_inject_noise,
            config.timestep_conditioning,
            &config.decoder_upsample_residual,
            &config.decoder_upsample_factor,
            vb.pp("decoder"),
        )?;

        let latents_mean = if vb.contains_tensor("latents_mean") {
            println!("Loading latents_mean from weights");
            vb.get(config.latent_channels, "latents_mean")?
        } else {
            Tensor::new(config.latents_mean.as_slice(), vb.device())?.to_dtype(vb.dtype())?
        };
        let latents_std = if vb.contains_tensor("latents_std") {
            println!("Loading latents_std from weights");
            vb.get(config.latent_channels, "latents_std")?
        } else {
            Tensor::new(config.latents_std.as_slice(), vb.device())?.to_dtype(vb.dtype())?
        };
        let vae_config = VaeConfig {
            scaling_factor: config.scaling_factor as f32,
            timestep_conditioning: config.timestep_conditioning,
        };

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            tile_sample_min_height: 512,
            tile_sample_min_width: 512,
            tile_sample_min_num_frames: 16,
            tile_sample_stride_height: 384,
            tile_sample_stride_width: 384,
            tile_sample_stride_num_frames: 8,
            scaling_factor: config.scaling_factor,
            spatial_compression_ratio: config.spatial_compression_ratio,
            temporal_compression_ratio: config.temporal_compression_ratio,
            use_slicing: false,
            use_tiling: true,
            use_framewise_encoding: false,
            use_framewise_decoding: true,
            num_sample_frames_batch_size: 1,
            num_latent_frames_batch_size: 1,
            config,
            latents_mean,
            latents_std,
            vae_config,
        })
    }

    pub fn enable_tiling(
        &mut self,
        tile_sample_min_height: Option<usize>,
        tile_sample_min_width: Option<usize>,
        tile_sample_min_num_frames: Option<usize>,
        tile_sample_stride_height: Option<usize>,
        tile_sample_stride_width: Option<usize>,
        tile_sample_stride_num_frames: Option<usize>,
    ) {
        self.use_tiling = true;
        if let Some(h) = tile_sample_min_height {
            self.tile_sample_min_height = h;
        }
        if let Some(w) = tile_sample_min_width {
            self.tile_sample_min_width = w;
        }
        if let Some(f) = tile_sample_min_num_frames {
            self.tile_sample_min_num_frames = f;
        }
        if let Some(h) = tile_sample_stride_height {
            self.tile_sample_stride_height = h;
        }
        if let Some(w) = tile_sample_stride_width {
            self.tile_sample_stride_width = w;
        }
        if let Some(f) = tile_sample_stride_num_frames {
            self.tile_sample_stride_num_frames = f;
        }
    }

    pub fn latents_mean(&self) -> &Tensor {
        &self.latents_mean
    }

    pub fn latents_std(&self) -> &Tensor {
        &self.latents_std
    }

    pub fn config(&self) -> &AutoencoderKLLtxVideoConfig {
        &self.config
    }

    pub fn vae_config(&self) -> &VaeConfig {
        &self.vae_config
    }

    fn split_batch_5d(x: &Tensor) -> Result<Vec<Tensor>> {
        let b = x.dims5()?.0;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            out.push(x.i((i..(i + 1), .., .., .., ..))?);
        }
        Ok(out)
    }

    fn blend_h(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        let blend = blend_extent.min(a.dims5()?.4).min(b.dims5()?.4);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, 1, 1, blend))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., .., .., 0..blend))?;
        let b_tail = b.i((.., .., .., .., blend..))?;

        let aw = a.dims5()?.4;
        let a_tail = a.i((.., .., .., .., (aw - blend)..aw))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 4)
    }

    fn blend_v(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        let blend = blend_extent.min(a.dims5()?.3).min(b.dims5()?.3);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, 1, blend, 1))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., .., 0..blend, ..))?;
        let b_tail = b.i((.., .., .., blend.., ..))?;

        let ah = a.dims5()?.3;
        let a_tail = a.i((.., .., .., (ah - blend)..ah, ..))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 3)
    }

    fn blend_t(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        let blend = blend_extent.min(a.dims5()?.2).min(b.dims5()?.2);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, blend, 1, 1))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., 0..blend, .., ..))?;
        let b_tail = b.i((.., .., blend.., .., ..))?;

        let at = a.dims5()?.2;
        let a_tail = a.i((.., .., (at - blend)..at, .., ..))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 2)
    }

    fn split_batch_2d(x: &Tensor) -> Result<Vec<Tensor>> {
        let (b, _d) = x.dims2()?;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            out.push(x.i((i..(i + 1), ..))?);
        }
        Ok(out)
    }

    fn encode_z(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let tile_sample_min_num_frames = self.tile_sample_min_num_frames;
        if self.use_framewise_encoding && x.dims5()?.2 > tile_sample_min_num_frames {
            return self.temporal_tiled_encode(x, train);
        }

        if self.use_tiling
            && (x.dims5()?.3 > self.tile_sample_min_height
                || x.dims5()?.4 > self.tile_sample_min_width)
        {
            return self.tiled_encode(x, train);
        }

        let mut h = self.encoder.forward(x, train)?;
        if let Some(ref qc) = self.quant_conv {
            h = qc.forward(&h)?;
        }
        Ok(h)
    }

    fn decode_z(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let model_dtype = self.decoder.conv_in.weight().dtype();
        let z = z.to_dtype(model_dtype)?;
        let temb_converted = match temb {
            Some(t) => Some(t.to_dtype(model_dtype)?),
            None => None,
        };

        let (_b, _c, t, h, w) = z.dims5()?;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;
        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;

        if self.use_framewise_decoding && t > tile_latent_min_t {
            let out = self.temporal_tiled_decode(&z, temb_converted.as_ref(), train)?;
            return Ok(out);
        }

        if self.use_tiling && (w > tile_latent_min_w || h > tile_latent_min_h) {
            let out = self.tiled_decode(&z, temb_converted.as_ref(), train)?;
            return Ok(out);
        }

        self.decoder.forward(&z, temb_converted.as_ref(), train)
    }

    pub fn encode(
        &self,
        x: &Tensor,
        return_dict: bool,
        train: bool,
    ) -> Result<(Option<AutoencoderKLOutput>, DiagonalGaussianDistribution)> {
        let h = if self.use_slicing && x.dims5()?.0 > 1 {
            let xs = Self::split_batch_5d(x)?;
            let mut encs = Vec::with_capacity(xs.len());
            for xs_i in xs.iter() {
                encs.push(self.encode_z(xs_i, train)?);
            }
            cat_dim(&encs, 0)?
        } else {
            self.encode_z(x, train)?
        };

        let posterior = DiagonalGaussianDistribution::new(&h)?;
        if return_dict {
            Ok((
                Some(AutoencoderKLOutput {
                    latent_dist: posterior.clone(),
                }),
                posterior,
            ))
        } else {
            Ok((None, posterior))
        }
    }

    pub fn decode(
        &self,
        z: &Tensor,
        temb: Option<&Tensor>,
        return_dict: bool,
        train: bool,
    ) -> Result<(Option<DecoderOutput>, Tensor)> {
        let decoded = if self.use_slicing && z.dims5()?.0 > 1 {
            let zs = Self::split_batch_5d(z)?;
            let ts = match temb {
                None => None,
                Some(t) => Some(Self::split_batch_2d(t)?),
            };

            let mut outs = Vec::with_capacity(zs.len());
            for (idx, z_i) in zs.iter().enumerate() {
                let t_i = ts.as_ref().map(|v| v[idx].as_ref());
                outs.push(self.decode_z(z_i, t_i, train)?);
            }
            cat_dim(&outs, 0)?
        } else {
            self.decode_z(z, temb, train)?
        };

        if return_dict {
            Ok((
                Some(DecoderOutput {
                    sample: decoded.clone(),
                }),
                decoded,
            ))
        } else {
            Ok((None, decoded))
        }
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        temb: Option<&Tensor>,
        sample_posterior: bool,
        return_dict: bool,
        train: bool,
    ) -> Result<(Option<DecoderOutput>, Tensor)> {
        let (_out, posterior) = self.encode(sample, true, train)?;
        let z = if sample_posterior {
            posterior.sample()?
        } else {
            posterior.mode()?
        };
        self.decode(&z, temb, return_dict, train)
    }

    fn tiled_encode(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (_b, _c, _t, height, width) = x.dims5()?;

        let latent_height = height / self.spatial_compression_ratio;
        let latent_width = width / self.spatial_compression_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_stride_h = self.tile_sample_stride_height / self.spatial_compression_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.spatial_compression_ratio;

        let blend_h = tile_latent_min_h.saturating_sub(tile_latent_stride_h);
        let blend_w = tile_latent_min_w.saturating_sub(tile_latent_stride_w);

        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(self.tile_sample_stride_height) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(self.tile_sample_stride_width) {
                let h_end = (i + self.tile_sample_min_height).min(height);
                let w_end = (j + self.tile_sample_min_width).min(width);
                let tile = x.i((.., .., .., i..h_end, j..w_end))?;
                let mut enc = self.encoder.forward(&tile, train)?;
                if let Some(ref qc) = self.quant_conv {
                    enc = qc.forward(&enc)?;
                }
                row.push(enc);
            }
            rows.push(row);
        }

        let mut prev_row_blended: Vec<Tensor> = Vec::new();
        let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
            let mut curr_row_blended: Vec<Tensor> = Vec::with_capacity(row.len());
            for (cj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();

                if ri > 0 {
                    let above = &prev_row_blended[cj];
                    tile = self.blend_v(above, &tile, blend_h)?;
                }
                if cj > 0 {
                    let left = &curr_row_blended[cj - 1];
                    tile = self.blend_h(left, &tile, blend_w)?;
                }

                curr_row_blended.push(tile.clone());

                let h_slice = tile_latent_stride_h.min(tile.dim(3)?);
                let w_slice = tile_latent_stride_w.min(tile.dim(4)?);
                let sliced_tile = tile.i((.., .., .., 0..h_slice, 0..w_slice))?;
                result_row.push(sliced_tile);
            }
            result_rows.push(cat_dim(&result_row, 4)?);
            prev_row_blended = curr_row_blended;
        }

        let enc = cat_dim(&result_rows, 3)?;
        enc.i((.., .., .., 0..latent_height, 0..latent_width))
    }

    fn tiled_decode(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let (_b, _c, _t, height, width) = z.dims5()?;

        let sample_height = height * self.spatial_compression_ratio;
        let sample_width = width * self.spatial_compression_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_stride_h = self.tile_sample_stride_height / self.spatial_compression_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.spatial_compression_ratio;

        let blend_h = self
            .tile_sample_min_height
            .saturating_sub(self.tile_sample_stride_height);
        let blend_w = self
            .tile_sample_min_width
            .saturating_sub(self.tile_sample_stride_width);

        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(tile_latent_stride_h) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(tile_latent_stride_w) {
                let h_end = (i + tile_latent_min_h).min(height);
                let w_end = (j + tile_latent_min_w).min(width);
                let tile = z.i((.., .., .., i..h_end, j..w_end))?;
                let dec = self.decoder.forward(&tile, temb, train)?;
                row.push(dec);
            }
            rows.push(row);
        }

        let mut prev_row_blended: Vec<Tensor> = Vec::new();
        let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
            let mut curr_row_blended: Vec<Tensor> = Vec::with_capacity(row.len());
            for (cj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();

                if ri > 0 {
                    let above = &prev_row_blended[cj];
                    tile = self.blend_v(above, &tile, blend_h)?;
                }
                if cj > 0 {
                    let left = &curr_row_blended[cj - 1];
                    tile = self.blend_h(left, &tile, blend_w)?;
                }

                curr_row_blended.push(tile.clone());

                let h_slice = self.tile_sample_stride_height.min(tile.dim(3)?);
                let w_slice = self.tile_sample_stride_width.min(tile.dim(4)?);
                let sliced_tile = tile.i((.., .., .., 0..h_slice, 0..w_slice))?;
                result_row.push(sliced_tile);
            }
            result_rows.push(cat_dim(&result_row, 4)?);
            prev_row_blended = curr_row_blended;
        }

        let dec = cat_dim(&result_rows, 3)?;
        dec.i((.., .., .., 0..sample_height, 0..sample_width))
    }

    fn temporal_tiled_encode(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (_b, _c, num_frames, _h, _w) = x.dims5()?;

        let latent_num_frames = (num_frames - 1) / self.temporal_compression_ratio + 1;

        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;
        let tile_latent_stride_t =
            self.tile_sample_stride_num_frames / self.temporal_compression_ratio;
        let blend_t = tile_latent_min_t.saturating_sub(tile_latent_stride_t);

        let mut row: Vec<Tensor> = Vec::new();
        for i in (0..num_frames).step_by(self.tile_sample_stride_num_frames) {
            let t_end = (i + self.tile_sample_min_num_frames + 1).min(num_frames);
            let tile = x.i((.., .., i..t_end, .., ..))?;

            let tile = if self.use_tiling
                && (tile.dims5()?.3 > self.tile_sample_min_height
                    || tile.dims5()?.4 > self.tile_sample_min_width)
            {
                self.tiled_encode(&tile, train)?
            } else {
                let mut h = self.encoder.forward(&tile, train)?;
                if let Some(ref qc) = self.quant_conv {
                    h = qc.forward(&h)?;
                }
                h
            };

            let tile = if i == 0 {
                tile.i((.., .., 1.., .., ..))?
            } else {
                tile
            };
            row.push(tile);
        }

        let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
        for (idx, tile) in row.iter().enumerate() {
            let tile = if idx > 0 {
                let blended = self.blend_t(&row[idx - 1], tile, blend_t)?;

                let end = tile_latent_stride_t.min(blended.dim(2)?);
                blended.i((.., .., 0..end, .., ..))?
            } else {
                let end = (tile_latent_stride_t + 1).min(tile.dim(2)?);
                tile.i((.., .., 0..end, .., ..))?
            };
            result_row.push(tile);
        }

        let enc = cat_dim(&result_row, 2)?;
        enc.i((.., .., 0..latent_num_frames, .., ..))
    }

    fn temporal_tiled_decode(
        &self,
        z: &Tensor,
        temb: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let (_b, _c, num_frames, _h, _w) = z.dims5()?;

        let num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;
        let tile_latent_stride_t =
            self.tile_sample_stride_num_frames / self.temporal_compression_ratio;

        let blend_t_sample = self
            .tile_sample_min_num_frames
            .saturating_sub(self.tile_sample_stride_num_frames);

        let mut row: Vec<Tensor> = Vec::new();
        for (loop_idx, i) in (0..num_frames).step_by(tile_latent_stride_t).enumerate() {
            let t_end = (i + tile_latent_min_t + 1).min(num_frames);
            let tile = z.i((.., .., i..t_end, .., ..))?;

            let decoded = if self.use_tiling
                && (tile.dims5()?.3 > tile_latent_min_h || tile.dims5()?.4 > tile_latent_min_w)
            {
                self.tiled_decode(&tile, temb, train)?
            } else {
                self.decoder.forward(&tile, temb, train)?
            };

            let decoded = if loop_idx > 0 {
                let t = decoded.dim(2)?;
                if t > 1 {
                    decoded.i((.., .., 0..(t - 1), .., ..))?
                } else {
                    decoded
                }
            } else {
                decoded
            };

            row.push(decoded);
        }

        let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
        for (idx, tile) in row.iter().enumerate() {
            let tile = if idx > 0 {
                let blended = self.blend_t(&row[idx - 1], tile, blend_t_sample)?;

                let end = self.tile_sample_stride_num_frames.min(blended.dim(2)?);
                blended.i((.., .., 0..end, .., ..))?
            } else {
                let end = (self.tile_sample_stride_num_frames + 1).min(tile.dim(2)?);
                tile.i((.., .., 0..end, .., ..))?
            };
            result_row.push(tile);
        }

        let dec = cat_dim(&result_row, 2)?;
        dec.i((.., .., 0..num_sample_frames, .., ..))
    }
}

impl VaeLtxVideo for AutoencoderKLLtxVideo {
    fn dtype(&self) -> DType {
        self.decoder.conv_in.weight().dtype()
    }
    fn spatial_compression_ratio(&self) -> usize {
        self.config.spatial_compression_ratio
    }
    fn temporal_compression_ratio(&self) -> usize {
        self.config.temporal_compression_ratio
    }
    fn config(&self) -> &VaeConfig {
        &self.vae_config
    }

    fn latents_mean(&self) -> &Tensor {
        &self.latents_mean
    }
    fn latents_std(&self) -> &Tensor {
        &self.latents_std
    }

    fn decode(&self, latents: &Tensor, timestep: Option<&Tensor>) -> Result<Tensor> {
        let (_, decoded) = self.decode(latents, timestep, false, false)?;
        Ok(decoded)
    }
}

impl AutoencoderMixin for AutoencoderKLLtxVideo {
    fn enable_tiling(&mut self) {
        AutoencoderKLLtxVideo::enable_tiling(self, None, None, None, None, None, None);
    }

    fn disable_tiling(&mut self) {
        self.use_tiling = false;
    }

    fn enable_slicing(&mut self) {
        self.use_slicing = true;
    }

    fn disable_slicing(&mut self) {
        self.use_slicing = false;
    }
}

impl VideoAutoencoder for AutoencoderKLLtxVideo {
    fn decode(&self, latents: &VideoLatents) -> std::result::Result<Tensor, AutoencoderError> {
        let latents = latents.to_canonical()?;
        let latents = latents.tensor.permute((0, 2, 1, 3, 4))?;
        let latents = latents.to_dtype(self.dtype())?;
        let (_, decoded) = AutoencoderKLLtxVideo::decode(self, &latents, None, false, false)?;
        Ok(decoded)
    }
}
