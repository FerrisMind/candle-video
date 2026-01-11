// Rust 2024 + candle
//
// Порт логики autoencoder_kl_wan.py на candle-тензорах.
// Важное: в исходнике много слоёв nn.Conv2d/nn.Conv3d/nn.Upsample/Dropout и т.д. [file:4]
// В candle-core нет готового "nn" набора как в torch, поэтому все свёртки/апсемплы
// вынесены в traits (Conv2dLike/Conv3dLike/Upsample2dLike), чтобы вы подключили
// реализацию из candle-nn или своей инфраструктуры весов. [file:4]
//
// Этот файл компилируем как модуль, но для полноценного запуска нужно предоставить реализации traits.

use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WanError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, WanError>;

fn ensure(cond: bool, msg: impl Into<String>) -> Result<()> {
    if cond { Ok(()) } else { Err(WanError::InvalidArgument(msg.into())) }
}

// -------------------- outputs / distributions --------------------

#[derive(Debug)]
pub struct DecoderOutput {
    pub sample: Tensor,
}

#[derive(Debug)]
pub struct AutoencoderKLOutput {
    pub latent_dist: DiagonalGaussianDistribution,
}

/// Аналог diffusers DiagonalGaussianDistribution(h) где h имеет 2*z_dim каналов (mean, logvar). [file:4]
#[derive(Debug, Clone)]
pub struct DiagonalGaussianDistribution {
    pub mean: Tensor,
    pub logvar: Tensor,
}

impl DiagonalGaussianDistribution {
    pub fn new(h: &Tensor, z_dim: usize) -> Result<Self> {
        // h: [b, 2*z_dim, t, h, w] [file:4]
        let (b, c, t, hh, ww) = h.dims5()?;
        ensure(c == 2 * z_dim, "expected 2*z_dim channels for posterior params")?;
        let mean = h.i((.., 0..z_dim as i64, .., .., ..))?;
        let logvar = h.i((.., z_dim as i64..(2 * z_dim) as i64, .., .., ..))?;
        Ok(Self { mean, logvar })
    }

    pub fn mode(&self) -> Tensor {
        self.mean.clone()
    }

    /// В оригинале sample(generator) — здесь оставляем интерфейс через trait Sampler. [file:4]
    pub fn sample(&self, sampler: &mut dyn GaussianSampler) -> Result<Tensor> {
        let eps = sampler.randn_like(&self.mean)?;
        // std = exp(0.5*logvar)
        let std = (self.logvar.to_dtype(DType::F32)? * 0.5f32)?.exp()?;
        Ok((&self.mean.to_dtype(DType::F32)? + (eps.to_dtype(DType::F32)? * std)?)?.to_dtype(self.mean.dtype())?)
    }
}

pub trait GaussianSampler {
    fn randn_like(&mut self, x: &Tensor) -> Result<Tensor>;
}

// -------------------- caching constants --------------------

const CACHE_T: usize = 2; // [file:4]

// -------------------- low-level ops: pad/patchify/unpatchify --------------------

fn pad_time_front(x: &Tensor, pad_t: usize) -> Result<Tensor> {
    // Python: F.pad(x, (0,0,0,0,pad_t,0)) meaning pad T dimension "front". [file:4]
    // x: [B,C,T,H,W]
    if pad_t == 0 {
        return Ok(x.clone());
    }
    let (b, c, t, h, w) = x.dims5()?;
    let zeros = Tensor::zeros((b, c, pad_t, h, w), x.dtype(), x.device())?;
    Tensor::cat(&[&zeros, x], 2).map_err(WanError::from)
}

pub fn patchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    // python patchify/unpatchify are pure reshapes+permutes for 2D patches, keeping time. [file:4]
    if patch_size == 1 {
        return Ok(x.clone());
    }
    let (b, c, f, h, w) = x.dims5()?;
    ensure(h % patch_size == 0 && w % patch_size == 0, "H and W must be divisible by patch_size")?;

    // view [b,c,f,h/ps,ps,w/ps,ps]
    let x = x.reshape((b, c, f, h / patch_size, patch_size, w / patch_size, patch_size))?;
    // permute (0,1,6,4,2,3,5)
    let x = x.permute((0, 1, 6, 4, 2, 3, 5))?;
    // view [b, c*ps*ps, f, h/ps, w/ps]
    Ok(x.reshape((b, c * patch_size * patch_size, f, h / patch_size, w / patch_size))?)
}

pub fn unpatchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return Ok(x.clone());
    }
    let (b, c_p, f, h, w) = x.dims5()?;
    let channels = c_p / (patch_size * patch_size);
    ensure(channels * patch_size * patch_size == c_p, "invalid channels for unpatchify")?;

    // view [b, c, ps, ps, f, h, w]
    let x = x.reshape((b, channels, patch_size, patch_size, f, h, w))?;
    // permute (0,1,4,5,3,6,2)
    let x = x.permute((0, 1, 4, 5, 3, 6, 2))?;
    // view [b,c,f,h*ps,w*ps]
    Ok(x.reshape((b, channels, f, h * patch_size, w * patch_size))?)
}

// -------------------- AvgDown3D / DupUp3D --------------------

#[derive(Clone, Debug)]
pub struct AvgDown3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    factor: usize,
    group_size: usize,
}

impl AvgDown3D {
    pub fn new(in_channels: usize, out_channels: usize, factor_t: usize, factor_s: usize) -> Result<Self> {
        let factor = factor_t * factor_s * factor_s;
        ensure((in_channels * factor) % out_channels == 0, "in_channels*factor must be divisible by out_channels")?;
        let group_size = in_channels * factor / out_channels;
        Ok(Self {
            in_channels,
            out_channels,
            factor_t,
            factor_s,
            factor,
            group_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // python:
        // pad_t = (factor_t - T%factor_t)%factor_t; pad time front; reshape/permute; mean over group. [file:4]
        let (b, c, t, h, w) = x.dims5()?;
        ensure(c == self.in_channels, "AvgDown3D channel mismatch")?;

        let pad_t = (self.factor_t - (t % self.factor_t)) % self.factor_t;
        let x = pad_time_front(x, pad_t)?;
        let (b, c, t2, h2, w2) = x.dims5()?;

        ensure(h2 % self.factor_s == 0 && w2 % self.factor_s == 0, "H/W must be divisible by factor_s")?;
        ensure(t2 % self.factor_t == 0, "T must be divisible by factor_t after pad")?;

        // view [B,C,T/f_t,f_t,H/f_s,f_s,W/f_s,f_s]
        let x = x.reshape((
            b,
            c,
            t2 / self.factor_t,
            self.factor_t,
            h2 / self.factor_s,
            self.factor_s,
            w2 / self.factor_s,
            self.factor_s,
        ))?;
        // permute (0,1,3,5,7,2,4,6)
        let x = x.permute((0, 1, 3, 5, 7, 2, 4, 6))?;
        // view [B, C*factor, T/f_t, H/f_s, W/f_s]
        let x = x.reshape((b, c * self.factor, t2 / self.factor_t, h2 / self.factor_s, w2 / self.factor_s))?;
        // view [B, out_channels, group_size, ...]
        let x = x.reshape((
            b,
            self.out_channels,
            self.group_size,
            t2 / self.factor_t,
            h2 / self.factor_s,
            w2 / self.factor_s,
        ))?;
        // mean over dim=2
        Ok(x.mean(2)?)
    }
}

#[derive(Clone, Debug)]
pub struct DupUp3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    factor: usize,
    repeats: usize,
}

impl DupUp3D {
    pub fn new(in_channels: usize, out_channels: usize, factor_t: usize, factor_s: usize) -> Result<Self> {
        let factor = factor_t * factor_s * factor_s;
        ensure((out_channels * factor) % in_channels == 0, "out_channels*factor must be divisible by in_channels")?;
        let repeats = out_channels * factor / in_channels;
        Ok(Self {
            in_channels,
            out_channels,
            factor_t,
            factor_s,
            factor,
            repeats,
        })
    }

    pub fn forward(&self, x: &Tensor, first_chunk: bool) -> Result<Tensor> {
        // python:
        // repeat_interleave(repeats, dim=1) then reshape/permute/unfold and optional crop for first_chunk. [file:4]
        let (b, c, t, h, w) = x.dims5()?;
        ensure(c == self.in_channels, "DupUp3D channel mismatch")?;

        let x = repeat_interleave_channels(x, self.repeats)?;
        // view [b, out_c, f_t, f_s, f_s, t, h, w]
        let x = x.reshape((
            b,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            t,
            h,
            w,
        ))?;
        // permute (0,1,5,2,6,3,7,4)
        let x = x.permute((0, 1, 5, 2, 6, 3, 7, 4))?;
        // view [b, out_c, t*f_t, h*f_s, w*f_s]
        let mut x = x.reshape((b, self.out_channels, t * self.factor_t, h * self.factor_s, w * self.factor_s))?;

        if first_chunk {
            // python: x = x[:, :, factor_t - 1 :, :, :] [file:4]
            let start = (self.factor_t - 1) as i64;
            let t3 = x.dims5()?.2 as i64;
            x = x.i((.., .., start..t3, .., ..))?;
        }
        Ok(x)
    }
}

fn repeat_interleave_channels(x: &Tensor, repeats: usize) -> Result<Tensor> {
    // x: [b,c,t,h,w] -> [b,c*repeats,t,h,w] by concat repeats copies along channel dim. [file:4]
    if repeats == 1 {
        return Ok(x.clone());
    }
    let mut parts = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        parts.push(x.clone());
    }
    Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?)
}

// -------------------- Conv/Upsample traits (to be implemented outside) --------------------

pub trait Conv3dLike: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub trait Conv2dLike: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub trait Upsample2dLike: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// WanCausalConv3d: в python это Conv3d + causal pad + optional cache concat in time. [file:4]
#[derive(Clone)]
pub struct WanCausalConv3d {
    pub conv: Arc<dyn Conv3dLike>,
    // python stores _padding = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, 2*pad_t, 0) [file:4]
    pub pad_t_total: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

impl WanCausalConv3d {
    pub fn new(conv: Arc<dyn Conv3dLike>, pad_t: usize, pad_h: usize, pad_w: usize) -> Self {
        Self {
            conv,
            pad_t_total: 2 * pad_t,
            pad_h,
            pad_w,
        }
    }

    pub fn forward(&self, x: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        // python:
        // if cache_x and pad_t_total>0: cat(cache_x,x) along time, then reduce pad_t_total by cache_x.t
        // pad as (w,w,h,h,pad_t_total,0) then conv. [file:4]
        let mut x2 = x.clone();
        let mut pad_t_total = self.pad_t_total;

        if let (Some(cx), true) = (cache_x, self.pad_t_total > 0) {
            let cx = cx.to_device(x.device())?;
            x2 = Tensor::cat(&[&cx, &x2], 2)?;
            let t_cached = cx.dims5()?.2;
            pad_t_total = pad_t_total.saturating_sub(t_cached);
        }

        // Candle doesn't have generic F.pad; implement minimal pad: time-front + spatial symmetric. [file:4]
        x2 = pad_time_front(&x2, pad_t_total)?;
        x2 = pad_hw_sym(&x2, self.pad_h, self.pad_w)?;
        self.conv.forward(&x2)
    }
}

fn pad_hw_sym(x: &Tensor, pad_h: usize, pad_w: usize) -> Result<Tensor> {
    if pad_h == 0 && pad_w == 0 {
        return Ok(x.clone());
    }
    // x: [b,c,t,h,w] -> pad h/w both sides with zeros.
    let (b, c, t, h, w) = x.dims5()?;
    let h2 = h + 2 * pad_h;
    let w2 = w + 2 * pad_w;
    let mut out = Tensor::zeros((b, c, t, h2, w2), x.dtype(), x.device())?;
    // copy to centered slice
    out = out
        .slice_assign(
            &[0..b, 0..c, 0..t, pad_h..(pad_h + h), pad_w..(pad_w + w)],
            x,
        )?;
    Ok(out)
}

// -------------------- WanRMS_norm / activations --------------------

#[derive(Clone)]
pub struct WanRmsNorm {
    channel_first: bool,
    scale: f32,
    gamma: Tensor,
    bias: Option<Tensor>,
}

impl WanRmsNorm {
    pub fn new(dim: usize, channel_first: bool, images: bool, bias: bool, device: &Device, dtype: DType) -> Result<Self> {
        // python: shape = (dim, *broadcastable_dims) if channel_first else (dim,) with broadcastable dims
        // images=True => (1,1) else (1,1,1). [file:4]
        let shape = if channel_first {
            if images { (dim, 1, 1) } else { (dim, 1, 1, 1) }
        } else {
            (dim, 1, 1) // not used in this file path; keep minimal
        };

        let gamma = Tensor::ones(shape, dtype, device)?;
        let bias_t = if bias { Some(Tensor::zeros(shape, dtype, device)?) } else { None };
        Ok(Self {
            channel_first,
            scale: (dim as f32).sqrt(),
            gamma,
            bias: bias_t,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // python: F.normalize(x, dim=(1 if channel_first else -1)) * scale * gamma + bias. [file:4]
        let x_f = x.to_dtype(DType::F32)?;
        let dim = if self.channel_first { 1 } else { x.rank() - 1 };
        let norm = x_f.sqr()?.sum_keepdim(dim)?.sqrt()?; // L2 norm
        let y = x_f.broadcast_div(&norm)?;
        let y = (y * self.scale)?;
        let y = y.broadcast_mul(&self.gamma.to_dtype(DType::F32)?)?;
        let y = if let Some(b) = &self.bias {
            y.broadcast_add(&b.to_dtype(DType::F32)?)?
        } else {
            y
        };
        Ok(y.to_dtype(x.dtype())?)
    }
}

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok((x * x.sigmoid()?)?)
}

// -------------------- WanResample --------------------

pub enum ResampleMode {
    None,
    Upsample2d,
    Upsample3d,
    Downsample2d,
    Downsample3d,
}

/// В python это комбо Upsample/Conv2d + time_conv для 3d режимов + кэш. [file:4]
pub struct WanResample {
    mode: ResampleMode,
    // resample path works on 2D frames: [b*t, c, h, w] -> conv2d
    upsample2d: Option<Arc<dyn Upsample2dLike>>,
    conv2d: Option<Arc<dyn Conv2dLike>>,
    time_conv: Option<WanCausalConv3d>,
}

impl WanResample {
    pub fn none() -> Self {
        Self { mode: ResampleMode::None, upsample2d: None, conv2d: None, time_conv: None }
    }

    pub fn upsample2d(ups: Arc<dyn Upsample2dLike>, conv: Arc<dyn Conv2dLike>) -> Self {
        Self { mode: ResampleMode::Upsample2d, upsample2d: Some(ups), conv2d: Some(conv), time_conv: None }
    }

    pub fn upsample3d(ups: Arc<dyn Upsample2dLike>, conv: Arc<dyn Conv2dLike>, time_conv: WanCausalConv3d) -> Self {
        Self { mode: ResampleMode::Upsample3d, upsample2d: Some(ups), conv2d: Some(conv), time_conv: Some(time_conv) }
    }

    pub fn downsample2d(conv: Arc<dyn Conv2dLike>) -> Self {
        Self { mode: ResampleMode::Downsample2d, upsample2d: None, conv2d: Some(conv), time_conv: None }
    }

    pub fn downsample3d(conv: Arc<dyn Conv2dLike>, time_conv: WanCausalConv3d) -> Self {
        Self { mode: ResampleMode::Downsample3d, upsample2d: None, conv2d: Some(conv), time_conv: Some(time_conv) }
    }

    pub fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize) -> Result<Tensor> {
        // x: [b,c,t,h,w] [file:4]
        let (b, c, t, h, w) = x.dims5()?;

        match self.mode {
            ResampleMode::None => Ok(x.clone()),

            ResampleMode::Upsample2d => {
                // python: permute to [b*t,c,h,w], upsample+conv2d, back. [file:4]
                let x2 = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
                let x2 = self.upsample2d.as_ref().unwrap().forward(&x2)?;
                let x2 = self.conv2d.as_ref().unwrap().forward(&x2)?;
                let (_, c2, h2, w2) = x2.dims4()?;
                Ok(x2.reshape((b, t, c2, h2, w2))?.permute((0, 2, 1, 3, 4))?)
            }

            ResampleMode::Downsample2d => {
                // python does ZeroPad2d then stride conv. Here conv2d is assumed to include that behavior. [file:4]
                let x2 = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
                let x2 = self.conv2d.as_ref().unwrap().forward(&x2)?;
                let (_, c2, h2, w2) = x2.dims4()?;
                Ok(x2.reshape((b, t, c2, h2, w2))?.permute((0, 2, 1, 3, 4))?)
            }

            ResampleMode::Upsample3d => {
                // python: time_conv produces [b,2,c,t,h,w] then stack => time doubled; then 2d upsample. [file:4]
                let tc = self.time_conv.as_ref().unwrap();
                let mut x_tc = x.clone();

                // cache handling mirrors python feat_cache semantics with "Rep" sentinel. [file:4]
                if let Some(cache) = feat_cache {
                    let idx = *feat_idx;
                    cache.ensure_len(idx + 1);
                    if cache[idx].is_empty() {
                        cache[idx] = FeatCache::Rep; // first call sentinel [file:4]
                        *feat_idx += 1;
                    } else {
                        let cache_x = build_cache_two_frames(x, &cache[idx])?;
                        let x_in = if matches!(cache[idx], FeatCache::Rep) {
                            tc.forward(&x_tc, None)?
                        } else {
                            tc.forward(&x_tc, cache[idx].as_tensor())?
                        };
                        cache[idx] = FeatCache::Tensor(cache_x);
                        *feat_idx += 1;
                        x_tc = x_in;
                    }
                } else {
                    x_tc = tc.forward(&x_tc, None)?;
                }

                // reshape b,2,c,t,h,w -> stack along time and reshape to [b,c,t*2,h,w] [file:4]
                let x_tc = x_tc.reshape((b, 2, c, t, h, w))?;
                let a = x_tc.i((.., 0, .., .., .., ..))?;
                let b2 = x_tc.i((.., 1, .., .., .., ..))?;
                let x_tc = Tensor::stack(&[&a, &b2], 3)?; // [b,c,t,2,h,w] but we want interleaved time
                let x_tc = x_tc.reshape((b, c, t * 2, h, w))?;

                // then 2d upsample+conv per frame
                let t2 = x_tc.dims5()?.2;
                let x2 = x_tc.permute((0, 2, 1, 3, 4))?.reshape((b * t2, c, h, w))?;
                let x2 = self.upsample2d.as_ref().unwrap().forward(&x2)?;
                let x2 = self.conv2d.as_ref().unwrap().forward(&x2)?;
                let (_, c2, h2, w2) = x2.dims4()?;
                Ok(x2.reshape((b, t2, c2, h2, w2))?.permute((0, 2, 1, 3, 4))?)
            }

            ResampleMode::Downsample3d => {
                // python: time_conv over cat([prev_last_frame, x]) and update cache. [file:4]
                let tc = self.time_conv.as_ref().unwrap();
                let mut x_tc = x.clone();

                if let Some(cache) = feat_cache {
                    let idx = *feat_idx;
                    cache.ensure_len(idx + 1);
                    if cache[idx].is_empty() {
                        cache[idx] = FeatCache::Tensor(x.clone());
                        *feat_idx += 1;
                        return Ok(x); // python stores cache then returns (next call does time_conv). [file:4]
                    } else {
                        let prev_last = cache[idx]
                            .as_tensor()
                            .ok_or_else(|| WanError::InvalidArgument("expected tensor cache".to_string()))?
                            .i((.., .., (cache[idx].as_tensor().unwrap().dims5()?.2 as i64 - 1).., .., ..))?;
                        let x_cat = Tensor::cat(&[&prev_last, &x_tc], 2)?;
                        let out = tc.forward(&x_cat, None)?;
                        let cache_x = x.i((.., .., (x.dims5()?.2 as i64 - 1).., .., ..))?;
                        cache[idx] = FeatCache::Tensor(cache_x);
                        *feat_idx += 1;
                        x_tc = out;
                    }
                } else {
                    // No cache: best-effort just apply time_conv on x. [file:4]
                    x_tc = tc.forward(&x_tc, None)?;
                }

                // after time downsample, do 2d downsample conv per frame
                let (b2, c2, t2, h2, w2) = x_tc.dims5()?;
                let x2 = x_tc.permute((0, 2, 1, 3, 4))?.reshape((b2 * t2, c2, h2, w2))?;
                let x2 = self.conv2d.as_ref().unwrap().forward(&x2)?;
                let (_, c3, h3, w3) = x2.dims4()?;
                Ok(x2.reshape((b2, t2, c3, h3, w3))?.permute((0, 2, 1, 3, 4))?)
            }
        }
    }
}

// -------------------- feature cache --------------------

#[derive(Clone)]
pub enum FeatCache {
    Empty,
    Rep,            // python sentinel "Rep" [file:4]
    Tensor(Tensor),
}

impl FeatCache {
    pub fn is_empty(&self) -> bool {
        matches!(self, FeatCache::Empty)
    }

    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            FeatCache::Tensor(t) => Some(t),
            _ => None,
        }
    }
}

pub trait FeatCacheVecExt {
    fn ensure_len(&mut self, n: usize);
}

impl FeatCacheVecExt for Vec<FeatCache> {
    fn ensure_len(&mut self, n: usize) {
        if self.len() < n {
            self.resize_with(n, || FeatCache::Empty);
        }
    }
}

fn build_cache_two_frames(x: &Tensor, cache_entry: &FeatCache) -> Result<Tensor> {
    // python caches last CACHE_T frames; if chunk too small (<2), prepend last frame from previous cache. [file:4]
    let (b, c, t, h, w) = x.dims5()?;
    let take = CACHE_T.min(t);
    let mut cache_x = x.i((.., .., (t as i64 - take as i64).., .., ..))?; // last frames

    if cache_x.dims5()?.2 < CACHE_T {
        if let Some(prev) = cache_entry.as_tensor() {
            let prev_last = prev.i((.., .., (prev.dims5()?.2 as i64 - 1).., .., ..))?;
            cache_x = Tensor::cat(&[&prev_last.to_device(x.device())?, &cache_x], 2)?;
        }
    }

    if cache_x.dims5()?.2 < CACHE_T {
        // python has special path for "Rep": prefix zeros. [file:4]
        let (b2, c2, t2, h2, w2) = cache_x.dims5()?;
        let zeros = Tensor::zeros((b2, c2, CACHE_T - t2, h2, w2), cache_x.dtype(), cache_x.device())?;
        cache_x = Tensor::cat(&[&zeros, &cache_x], 2)?;
    }

    Ok(cache_x)
}

// -------------------- WanResidualBlock / WanAttentionBlock / MidBlock --------------------
// Для полного VAE нужны Conv2d/Conv3d веса + attention (single-head) на 2D фичах. [file:4]
// Здесь оставим forward-совместимые структуры, но сами свёртки/Conv2dLike/Conv3dLike должны прийти извне.

pub struct WanResidualBlock {
    norm1: WanRmsNorm,
    conv1: WanCausalConv3d,
    norm2: WanRmsNorm,
    // dropout: no-op in inference path is acceptable; keep as placeholder. [file:4]
    conv2: WanCausalConv3d,
    conv_shortcut: Option<WanCausalConv3d>,
}

impl WanResidualBlock {
    pub fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize) -> Result<Tensor> {
        // h = shortcut(x) [file:4]
        let h = if let Some(sc) = &self.conv_shortcut {
            sc.forward(x, None)?
        } else {
            x.clone()
        };

        // norm+act
        let mut y = self.norm1.forward(x)?;
        y = silu(&y)?;

        // conv1 with cache
        y = causal_conv_cached(&self.conv1, &y, feat_cache, feat_idx)?;

        // norm+act
        y = self.norm2.forward(&y)?;
        y = silu(&y)?;

        // conv2 with cache
        y = causal_conv_cached(&self.conv2, &y, feat_cache, feat_idx)?;

        Ok((y + h)?)
    }
}

fn causal_conv_cached(
    conv: &WanCausalConv3d,
    x: &Tensor,
    feat_cache: Option<&mut Vec<FeatCache>>,
    feat_idx: &mut usize,
) -> Result<Tensor> {
    if let Some(cache) = feat_cache {
        let idx = *feat_idx;
        cache.ensure_len(idx + 1);
        let cache_x = build_cache_two_frames(x, &cache[idx])?;
        let out = conv.forward(x, cache[idx].as_tensor())?;
        cache[idx] = FeatCache::Tensor(cache_x);
        *feat_idx += 1;
        Ok(out)
    } else {
        conv.forward(x, None)
    }
}

pub struct WanAttentionBlock {
    // python uses Conv2d to_qkv and proj and SDPA with one head. [file:4]
    norm: WanRmsNorm,
    to_qkv: Arc<dyn Conv2dLike>,
    proj: Arc<dyn Conv2dLike>,
}

impl WanAttentionBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let (b, c, t, h, w) = x.dims5()?;

        // [b*t, c, h, w]
        let mut y = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
        y = self.norm.forward(&y)?; // norm expects images=True channels-first, ok for 4D [file:4]

        let qkv = self.to_qkv.forward(&y)?; // [b*t, 3c, h, w]
        // reshape -> [b*t, 1, hw, 3c] then chunk -> q,k,v each [b*t,1,hw,c] [file:4]
        let hw = h * w;
        let qkv = qkv.reshape((b * t, 1, 3 * c, hw))?.permute((0, 1, 3, 2))?; // [bt,1,hw,3c]
        let q = qkv.i((.., .., .., 0..c as i64))?;
        let k = qkv.i((.., .., .., c as i64..(2 * c) as i64))?;
        let v = qkv.i((.., .., .., (2 * c) as i64..(3 * c) as i64))?;

        let y = sdpa_1head(&q, &k, &v)?; // [bt,1,hw,c]
        let y = y.squeeze(1)?.permute((0, 2, 1))?.reshape((b * t, c, h, w))?;
        let y = self.proj.forward(&y)?;

        let y = y.reshape((b, t, c, h, w))?.permute((0, 2, 1, 3, 4))?;
        Ok((y + identity)?)
    }
}

fn sdpa_1head(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // q,k,v: [bt, 1, hw, c] [file:4]
    let (_bt, _h, hw, c) = q.dims4()?;
    let scale = 1f32 / (c as f32).sqrt();
    let k_t = k.transpose(2, 3)?; // [bt,1,c,hw]
    let scores = (q.matmul(&k_t)? * scale)?; // [bt,1,hw,hw]
    let probs = scores.softmax(3)?;
    Ok(probs.matmul(v)?)
}

pub struct WanMidBlock {
    resnets: Vec<WanResidualBlock>,
    attentions: Vec<WanAttentionBlock>,
}

impl WanMidBlock {
    pub fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize) -> Result<Tensor> {
        // python: resnet0, then for each attn/resnet pair. [file:4]
        let mut y = self.resnets[0].forward(x, feat_cache, feat_idx)?;
        for (attn, resnet) in self.attentions.iter().zip(self.resnets.iter().skip(1)) {
            y = attn.forward(&y)?;
            y = resnet.forward(&y, feat_cache, feat_idx)?;
        }
        Ok(y)
    }
}

// -------------------- Encoder/Decoder skeletons --------------------

pub struct WanEncoder3d {
    conv_in: WanCausalConv3d,
    down_blocks: Vec<Box<dyn EncoderLayer3d>>,
    mid_block: WanMidBlock,
    norm_out: WanRmsNorm,
    conv_out: WanCausalConv3d,
}

pub trait EncoderLayer3d: Send + Sync {
    fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize) -> Result<Tensor>;
}

impl WanEncoder3d {
    pub fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize) -> Result<Tensor> {
        // conv_in with cache like python. [file:4]
        let mut y = causal_conv_cached(&self.conv_in, x, feat_cache, feat_idx)?;
        for layer in &self.down_blocks {
            y = layer.forward(&y, feat_cache, feat_idx)?;
        }
        y = self.mid_block.forward(&y, feat_cache, feat_idx)?;
        y = self.norm_out.forward(&y)?;
        y = silu(&y)?;
        y = causal_conv_cached(&self.conv_out, &y, feat_cache, feat_idx)?;
        Ok(y)
    }
}

pub struct WanDecoder3d {
    conv_in: WanCausalConv3d,
    mid_block: WanMidBlock,
    up_blocks: Vec<Box<dyn DecoderLayer3d>>,
    norm_out: WanRmsNorm,
    conv_out: WanCausalConv3d,
}

pub trait DecoderLayer3d: Send + Sync {
    fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize, first_chunk: bool) -> Result<Tensor>;
}

impl WanDecoder3d {
    pub fn forward(&self, x: &Tensor, feat_cache: Option<&mut Vec<FeatCache>>, feat_idx: &mut usize, first_chunk: bool) -> Result<Tensor> {
        let mut y = causal_conv_cached(&self.conv_in, x, feat_cache, feat_idx)?;
        y = self.mid_block.forward(&y, feat_cache, feat_idx)?;
        for blk in &self.up_blocks {
            y = blk.forward(&y, feat_cache, feat_idx, first_chunk)?;
        }
        y = self.norm_out.forward(&y)?;
        y = silu(&y)?;
        y = causal_conv_cached(&self.conv_out, &y, feat_cache, feat_idx)?;
        Ok(y)
    }
}

// -------------------- AutoencoderKLWan --------------------

#[derive(Clone, Debug)]
pub struct AutoencoderKLWanConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub patch_size: Option<usize>,
    pub scale_factor_temporal: usize,
    pub scale_factor_spatial: usize,

    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

pub struct AutoencoderKLWan {
    pub cfg: AutoencoderKLWanConfig,

    pub encoder: WanEncoder3d,
    pub quant_conv: WanCausalConv3d,       // 1x1 causal conv3d [file:4]
    pub post_quant_conv: WanCausalConv3d,  // 1x1 causal conv3d [file:4]
    pub decoder: WanDecoder3d,

    pub use_slicing: bool,
    pub use_tiling: bool,

    pub tile_sample_min_height: usize,
    pub tile_sample_min_width: usize,
    pub tile_sample_stride_height: usize,
    pub tile_sample_stride_width: usize,

    // cached conv feature maps for causal conv inference
    conv_num: usize,
    conv_idx: usize,
    feat_map: Vec<FeatCache>,

    enc_conv_num: usize,
    enc_conv_idx: usize,
    enc_feat_map: Vec<FeatCache>,
}

impl AutoencoderKLWan {
    pub fn z_dim(&self) -> usize { self.cfg.z_dim }
    pub fn latents_mean(&self) -> &[f32] { &self.cfg.latents_mean }
    pub fn latents_std(&self) -> &[f32] { &self.cfg.latents_std }
    pub fn scale_factor_temporal(&self) -> usize { self.cfg.scale_factor_temporal }
    pub fn scale_factor_spatial(&self) -> usize { self.cfg.scale_factor_spatial }
    pub fn dtype(&self) -> DType { DType::F32 } // в python зависит от параметров; для VAE часто fp16/bf16. [file:4]

    pub fn enable_tiling(
        &mut self,
        tile_sample_min_height: Option<usize>,
        tile_sample_min_width: Option<usize>,
        tile_sample_stride_height: Option<usize>,
        tile_sample_stride_width: Option<usize>,
    ) {
        self.use_tiling = true;
        if let Some(v) = tile_sample_min_height { self.tile_sample_min_height = v; }
        if let Some(v) = tile_sample_min_width { self.tile_sample_min_width = v; }
        if let Some(v) = tile_sample_stride_height { self.tile_sample_stride_height = v; }
        if let Some(v) = tile_sample_stride_width { self.tile_sample_stride_width = v; }
    }

    pub fn clear_cache(&mut self) {
        // python precomputes conv counts; here we keep them set by constructor. [file:4]
        self.conv_idx = 0;
        self.feat_map.resize_with(self.conv_num, || FeatCache::Empty);

        self.enc_conv_idx = 0;
        self.enc_feat_map.resize_with(self.enc_conv_num, || FeatCache::Empty);
    }

    fn encode_impl(&mut self, x: &Tensor) -> Result<Tensor> {
        // python: optional patchify; optional tiled_encode; else chunked temporal encode with stride 4. [file:4]
        let (_, _, num_frames, height, width) = x.dims5()?;
        self.clear_cache();

        let mut x2 = x.clone();
        if let Some(ps) = self.cfg.patch_size {
            x2 = patchify(&x2, ps)?;
        }

        if self.use_tiling && (width > self.tile_sample_min_width || height > self.tile_sample_min_height) {
            return self.tiled_encode(&x2);
        }

        let iter = 1 + (num_frames - 1) / 4;
        let mut out_opt: Option<Tensor> = None;

        for i in 0..iter {
            self.enc_conv_idx = 0;
            let chunk = if i == 0 {
                x2.i((.., .., 0..1, .., ..))?
            } else {
                let start = 1 + 4 * (i - 1);
                let end = 1 + 4 * i;
                x2.i((.., .., start as i64..end as i64, .., ..))?
            };
            let chunk_out = self.encoder.forward(&chunk, Some(&mut self.enc_feat_map), &mut self.enc_conv_idx)?;
            out_opt = Some(match out_opt {
                None => chunk_out,
                Some(prev) => Tensor::cat(&[&prev, &chunk_out], 2)?,
            });
        }

        let out = out_opt.ok_or_else(|| WanError::InvalidArgument("empty encode".to_string()))?;
        let enc = self.quant_conv.forward(&out, None)?;
        self.clear_cache();
        Ok(enc)
    }

    pub fn encode(&mut self, x: &Tensor, return_dict: bool) -> Result<std::result::Result<AutoencoderKLOutput, (DiagonalGaussianDistribution,)>> {
        // python supports use_slicing across batch. [file:4]
        let b = x.dims5()?.0;
        let h = if self.use_slicing && b > 1 {
            let mut slices = Vec::with_capacity(b);
            for i in 0..b {
                let xs = x.i((i as i64..(i as i64 + 1), .., .., .., ..))?;
                slices.push(self.encode_impl(&xs)?);
            }
            Tensor::cat(&slices.iter().collect::<Vec<_>>(), 0)?
        } else {
            self.encode_impl(x)?
        };

        let posterior = DiagonalGaussianDistribution::new(&h, self.cfg.z_dim)?;
        if return_dict {
            Ok(Ok(AutoencoderKLOutput { latent_dist: posterior }))
        } else {
            Ok(Err((posterior,)))
        }
    }

    fn decode_impl(&mut self, z: &Tensor) -> Result<Tensor> {
        // python: optional tiled_decode; else per-frame decode using feat_cache + first_chunk flag. [file:4]
        let (_, _, num_frames, height, width) = z.dims5()?;

        let tile_latent_min_h = self.tile_sample_min_height / self.cfg.scale_factor_spatial;
        let tile_latent_min_w = self.tile_sample_min_width / self.cfg.scale_factor_spatial;
        if self.use_tiling && (width > tile_latent_min_w || height > tile_latent_min_h) {
            return self.tiled_decode(z);
        }

        self.clear_cache();
        let x = self.post_quant_conv.forward(z, None)?;

        let mut out_opt: Option<Tensor> = None;
        for i in 0..num_frames {
            self.conv_idx = 0;
            let frame = x.i((.., .., i as i64..(i as i64 + 1), .., ..))?;
            let decoded = self.decoder.forward(
                &frame,
                Some(&mut self.feat_map),
                &mut self.conv_idx,
                i == 0,
            )?;
            out_opt = Some(match out_opt {
                None => decoded,
                Some(prev) => Tensor::cat(&[&prev, &decoded], 2)?,
            });
        }
        let mut out = out_opt.ok_or_else(|| WanError::InvalidArgument("empty decode".to_string()))?;

        if let Some(ps) = self.cfg.patch_size {
            out = unpatchify(&out, ps)?;
        }
        out = out.clamp(-1f32, 1f32)?;

        self.clear_cache();
        Ok(out)
    }

    pub fn decode(&mut self, z: &Tensor, return_dict: bool) -> Result<std::result::Result<DecoderOutput, (Tensor,)>> {
        let b = z.dims5()?.0;
        let decoded = if self.use_slicing && b > 1 {
            let mut slices = Vec::with_capacity(b);
            for i in 0..b {
                let zs = z.i((i as i64..(i as i64 + 1), .., .., .., ..))?;
                slices.push(self.decode_impl(&zs)?);
            }
            Tensor::cat(&slices.iter().collect::<Vec<_>>(), 0)?
        } else {
            self.decode_impl(z)?
        };

        if return_dict {
            Ok(Ok(DecoderOutput { sample: decoded }))
        } else {
            Ok(Err((decoded,)))
        }
    }

    pub fn forward(
        &mut self,
        sample: &Tensor,
        sample_posterior: bool,
        return_dict: bool,
        sampler: Option<&mut dyn GaussianSampler>,
    ) -> Result<std::result::Result<DecoderOutput, (Tensor,)>> {
        // python: posterior = encode(...).latent_dist; z = sample or mode; dec = decode(z). [file:4]
        let enc = self.encode(sample, true)?;
        let posterior = match enc {
            Ok(out) => out.latent_dist,
            Err(_) => return Err(WanError::InvalidArgument("encode must return dict in forward".to_string())),
        };

        let z = if sample_posterior {
            let s = sampler.ok_or_else(|| WanError::InvalidArgument("sampler required when sample_posterior=true".to_string()))?;
            posterior.sample(s)?
        } else {
            posterior.mode()
        };

        let dec = self.decode(&z, return_dict)?;
        Ok(match dec {
            Ok(out) => Ok(out),
            Err((t,)) => Err((t,)),
        })
    }

    // ---------------- tiling + blending ----------------

    fn blend_v(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python blends along height dimension near boundary. [file:4]
        let blend_extent = blend_extent.min(a.dims5()?.3).min(b.dims5()?.3);
        if blend_extent == 0 { return Ok(b.clone()); }

        // vectorized: b[..., 0:be, :] = a[..., -be:, :]*(1-w) + b[..., 0:be,:]*w
        let be = blend_extent;
        let a_tail = a.i((.., .., .., (a.dims5()?.3 as i64 - be as i64).., ..))?; // last be rows
        let b_head = b.i((.., .., .., 0..be as i64, ..))?;

        let w = ramp(be, b.device())?.reshape((1, 1, 1, be, 1))?; // [1,1,1,be,1]
        let one = Tensor::ones((), DType::F32, b.device())?;
        let out_head = (&a_tail.to_dtype(DType::F32)? * (&one - &w)?)? + (&b_head.to_dtype(DType::F32)? * &w)?;
        let out_head = out_head.to_dtype(b.dtype())?;

        let b_rest = b.i((.., .., .., be as i64.., ..))?;
        Ok(Tensor::cat(&[&out_head, &b_rest], 3)?)
    }

    fn blend_h(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python blends along width dimension near boundary. [file:4]
        let blend_extent = blend_extent.min(a.dims5()?.4).min(b.dims5()?.4);
        if blend_extent == 0 { return Ok(b.clone()); }

        let be = blend_extent;
        let a_tail = a.i((.., .., .., .., (a.dims5()?.4 as i64 - be as i64)..))?;
        let b_head = b.i((.., .., .., .., 0..be as i64))?;

        let w = ramp(be, b.device())?.reshape((1, 1, 1, 1, be))?;
        let one = Tensor::ones((), DType::F32, b.device())?;
        let out_head = (&a_tail.to_dtype(DType::F32)? * (&one - &w)?)? + (&b_head.to_dtype(DType::F32)? * &w)?;
        let out_head = out_head.to_dtype(b.dtype())?;

        let b_rest = b.i((.., .., .., .., be as i64..))?;
        Ok(Tensor::cat(&[&out_head, &b_rest], 4)?)
    }

    fn tiled_encode(&mut self, x: &Tensor) -> Result<Tensor> {
        // Прямая адаптация python tiled_encode, но без циклов по blend (векторизовано blend_v/h). [file:4]
        let (_, _, num_frames, height, width) = x.dims5()?;

        let mut encode_ratio = self.cfg.scale_factor_spatial;
        if let Some(ps) = self.cfg.patch_size {
            ensure(encode_ratio % ps == 0, "scale_factor_spatial must be divisible by patch_size")?;
            encode_ratio /= ps;
        }

        let latent_h = height / encode_ratio;
        let latent_w = width / encode_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / encode_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / encode_ratio;
        let tile_latent_stride_h = self.tile_sample_stride_height / encode_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / encode_ratio;

        let blend_h = tile_latent_min_h - tile_latent_stride_h;
        let blend_w = tile_latent_min_w - tile_latent_stride_w;

        // tiles -> rows[i][j] each tile is [b, 2*z_dim, t_lat, h_lat, w_lat] [file:4]
        let mut rows: Vec<Vec<Tensor>> = Vec::new();

        for i in (0..height).step_by(self.tile_sample_stride_height) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(self.tile_sample_stride_width) {
                self.clear_cache();
                let frame_range = 1 + (num_frames - 1) / 4;

                let mut time_chunks: Vec<Tensor> = Vec::new();
                for k in 0..frame_range {
                    self.enc_conv_idx = 0;
                    let tile = if k == 0 {
                        x.i((.., .., 0..1, i as i64..(i + self.tile_sample_min_height) as i64, j as i64..(j + self.tile_sample_min_width) as i64))?
                    } else {
                        let start = 1 + 4 * (k - 1);
                        let end = 1 + 4 * k;
                        x.i((.., .., start as i64..end as i64, i as i64..(i + self.tile_sample_min_height) as i64, j as i64..(j + self.tile_sample_min_width) as i64))?
                    };
                    let tile = self.encoder.forward(&tile, Some(&mut self.enc_feat_map), &mut self.enc_conv_idx)?;
                    let tile = self.quant_conv.forward(&tile, None)?;
                    time_chunks.push(tile);
                }

                let tile_lat = Tensor::cat(&time_chunks.iter().collect::<Vec<_>>(), 2)?;
                row.push(tile_lat);
            }
            rows.push(row);
        }

        self.clear_cache();

        // blend + stitch
        let mut result_rows: Vec<Tensor> = Vec::new();
        for i in 0..rows.len() {
            let mut stitched: Vec<Tensor> = Vec::new();
            for j in 0..rows[i].len() {
                let mut tile = rows[i][j].clone();
                if i > 0 {
                    tile = self.blend_v(&rows[i - 1][j], &tile, blend_h)?;
                }
                if j > 0 {
                    tile = self.blend_h(&rows[i][j - 1], &tile, blend_w)?;
                }

                // take stride region
                tile = tile.i((.., .., .., 0..tile_latent_stride_h as i64, 0..tile_latent_stride_w as i64))?;
                stitched.push(tile);
            }
            let row_cat = Tensor::cat(&stitched.iter().collect::<Vec<_>>(), 4)?;
            result_rows.push(row_cat);
        }

        let enc = Tensor::cat(&result_rows.iter().collect::<Vec<_>>(), 3)?;
        Ok(enc.i((.., .., .., 0..latent_h as i64, 0..latent_w as i64))?)
    }

    fn tiled_decode(&mut self, z: &Tensor) -> Result<Tensor> {
        // Прямая адаптация python tiled_decode. [file:4]
        let (_, _, num_frames, height, width) = z.dims5()?;
        let mut sample_h = height * self.cfg.scale_factor_spatial;
        let mut sample_w = width * self.cfg.scale_factor_spatial;

        let tile_latent_min_h = self.tile_sample_min_height / self.cfg.scale_factor_spatial;
        let tile_latent_min_w = self.tile_sample_min_width / self.cfg.scale_factor_spatial;
        let tile_latent_stride_h = self.tile_sample_stride_height / self.cfg.scale_factor_spatial;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.cfg.scale_factor_spatial;

        let (tile_sample_stride_h, tile_sample_stride_w, blend_h, blend_w) = if let Some(ps) = self.cfg.patch_size {
            sample_h /= ps;
            sample_w /= ps;
            (
                self.tile_sample_stride_height / ps,
                self.tile_sample_stride_width / ps,
                self.tile_sample_min_height / ps - self.tile_sample_stride_height / ps,
                self.tile_sample_min_width / ps - self.tile_sample_stride_width / ps,
            )
        } else {
            (
                self.tile_sample_stride_height,
                self.tile_sample_stride_width,
                self.tile_sample_min_height - self.tile_sample_stride_height,
                self.tile_sample_min_width - self.tile_sample_stride_width,
            )
        };

        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(tile_latent_stride_h) {
            let mut row = Vec::new();
            for j in (0..width).step_by(tile_latent_stride_w) {
                self.clear_cache();
                let mut time = Vec::new();

                for k in 0..num_frames {
                    self.conv_idx = 0;
                    let tile = z.i((.., .., k as i64..(k as i64 + 1), i as i64..(i + tile_latent_min_h) as i64, j as i64..(j + tile_latent_min_w) as i64))?;
                    let tile = self.post_quant_conv.forward(&tile, None)?;
                    let decoded = self.decoder.forward(&tile, Some(&mut self.feat_map), &mut self.conv_idx, k == 0)?;
                    time.push(decoded);
                }

                row.push(Tensor::cat(&time.iter().collect::<Vec<_>>(), 2)?);
            }
            rows.push(row);
        }

        self.clear_cache();

        let mut result_rows: Vec<Tensor> = Vec::new();
        for i in 0..rows.len() {
            let mut result_row = Vec::new();
            for j in 0..rows[i].len() {
                let mut tile = rows[i][j].clone();
                if i > 0 { tile = self.blend_v(&rows[i - 1][j], &tile, blend_h)?; }
                if j > 0 { tile = self.blend_h(&rows[i][j - 1], &tile, blend_w)?; }

                tile = tile.i((.., .., .., 0..tile_sample_stride_h as i64, 0..tile_sample_stride_w as i64))?;
                result_row.push(tile);
            }
            result_rows.push(Tensor::cat(&result_row.iter().collect::<Vec<_>>(), 4)?);
        }

        let mut dec = Tensor::cat(&result_rows.iter().collect::<Vec<_>>(), 3)?;
        dec = dec.i((.., .., .., 0..sample_h as i64, 0..sample_w as i64))?;

        if let Some(ps) = self.cfg.patch_size {
            dec = unpatchify(&dec, ps)?;
        }
        Ok(dec.clamp(-1f32, 1f32)?)
    }
}

fn ramp(n: usize, device: &Device) -> Result<Tensor> {
    // w[i] = i/n
    if n == 0 {
        return Ok(Tensor::zeros((0,), DType::F32, device)?);
    }
    let mut v = Vec::<f32>::with_capacity(n);
    for i in 0..n {
        v.push(i as f32 / (n as f32));
    }
    Ok(Tensor::from_vec(v, (n,), device)?)
}
