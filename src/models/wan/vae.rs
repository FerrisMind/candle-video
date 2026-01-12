//! AutoencoderKLWan - 3D VAE for Wan video generation.
//!
//! Port of diffusers AutoencoderKLWan to Candle with VarBuilder.
//! Supports Wan 2.1 (standard) and Wan 2.2 (residual) architectures.

#![allow(clippy::needless_option_as_deref)]

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::interfaces::distributions::DiagonalGaussianDistribution;
use crate::models::wan::config::AutoencoderKLWanConfig;
use crate::ops::conv3d::{Conv3d, Conv3dConfig, PaddingMode};

// =============================================================================
// Constants
// =============================================================================

const CACHE_T: usize = 2;

// =============================================================================
// Utility Functions
// =============================================================================

/// Patchify video tensor: [B, C, F, H, W] -> [B, C*ps*ps, F, H/ps, W/ps]
pub fn patchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return Ok(x.clone());
    }
    let (b, c, f, h, w) = x.dims5()?;
    if h % patch_size != 0 || w % patch_size != 0 {
        candle_core::bail!(
            "Height ({}) and width ({}) must be divisible by patch_size ({})",
            h,
            w,
            patch_size
        );
    }

    let hp = h / patch_size;
    let wp = w / patch_size;
    let ps = patch_size;

    // Step 1: Reshape to [b, c, f, hp, ps, wp*ps] - merge w dimension
    let x = x.reshape(&[b, c, f, hp, ps, wp * ps])?;
    // Step 2: Reshape to [b, c, f, hp, ps, wp, ps]
    let x = x.reshape(&[b, c * f, hp, ps, wp, ps])?;
    // Step 3: Permute to [b, c*f, ps, ps, hp, wp] = [b, cf, ps, ps, hp, wp]
    let x = x.permute([0, 1, 5, 3, 2, 4])?;
    // Step 4: Reshape to [b, c, f, ps*ps, hp, wp]
    let x = x.reshape(&[b, c, f, ps * ps, hp, wp])?;
    // Step 5: Permute to [b, c, ps*ps, f, hp, wp]
    let x = x.permute([0, 1, 3, 2, 4, 5])?;
    // Step 6: Reshape to [b, c*ps*ps, f, hp, wp]
    x.reshape(&[b, c * ps * ps, f, hp, wp])
}

/// Unpatchify video tensor: [B, C*ps*ps, F, H, W] -> [B, C, F, H*ps, W*ps]
pub fn unpatchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return Ok(x.clone());
    }
    let (b, c_p, f, hp, wp) = x.dims5()?;
    let ps = patch_size;
    let channels = c_p / (ps * ps);
    if channels * ps * ps != c_p {
        candle_core::bail!("Invalid channels for unpatchify: {}", c_p);
    }

    // Step 1: Reshape to [b, c, ps*ps, f, hp, wp]
    let x = x.reshape(&[b, channels, ps * ps, f, hp, wp])?;
    // Step 2: Permute to [b, c, f, ps*ps, hp, wp]
    let x = x.permute([0, 1, 3, 2, 4, 5])?;
    // Step 3: Reshape to [b, c*f, ps, ps, hp, wp]
    let x = x.reshape(&[b, channels * f, ps, ps, hp, wp])?;
    // Step 4: Permute to [b, c*f, hp, ps, wp, ps]
    let x = x.permute([0, 1, 4, 2, 5, 3])?;
    // Step 5: Reshape to [b, c, f, hp*ps, wp*ps]
    x.reshape(&[b, channels, f, hp * ps, wp * ps])
}

fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

fn pad_time_front(x: &Tensor, pad_t: usize) -> Result<Tensor> {
    if pad_t == 0 {
        return Ok(x.clone());
    }
    // OPTIMIZATION: Use Candle's built-in pad_with_zeros instead of zeros + cat
    // This is more memory efficient as it avoids creating intermediate tensors
    x.pad_with_zeros(2, pad_t, 0)
}

// Slice 5D tensor along time dimension: x[:, :, start:end, :, :]
fn slice_time(x: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    x.narrow(2, start, len)
}

// =============================================================================
// WanRmsNorm - RMS Normalization
// =============================================================================

/// RMS normalization for Wan VAE (channel-first, 5D tensors).
#[derive(Debug, Clone)]
pub struct WanRmsNorm {
    gamma: Tensor,
    scale: f32,
    #[allow(dead_code)]
    images: bool, // true for 4D (images), false for 5D (video)
}

impl WanRmsNorm {
    pub fn new(dim: usize, images: bool, vb: VarBuilder) -> Result<Self> {
        let shape: Vec<usize> = if images {
            vec![dim, 1, 1]
        } else {
            vec![dim, 1, 1, 1]
        };
        let gamma = vb.get_with_hints(shape.as_slice(), "gamma", candle_nn::Init::Const(1.0))?;
        Ok(Self {
            gamma,
            scale: (dim as f32).sqrt(),
            images,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        // L2 normalize along channel dimension (dim=1)
        // Work in native dtype to save memory
        let norm = x
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-6f64, 1e6f64)?;  // Use f64 for clamp bounds
        let y = x.broadcast_div(&norm)?;
        let y = (y * self.scale as f64)?;
        let gamma = self.gamma.to_dtype(dtype)?;
        y.broadcast_mul(&gamma)
    }
}

// =============================================================================
// WanCausalConv3d - Causal 3D Convolution (now using native Conv3d)
// =============================================================================

/// Causal 3D convolution - now a type alias to native Conv3d.
/// Kept for backward compatibility.
pub type WanCausalConv3d = Conv3d;

/// Helper function to create a WanCausalConv3d with the expected parameters.
/// This matches the old WanCausalConv3d::new() signature.
pub fn wan_causal_conv3d_new(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize), // (t, h, w)
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    vb: VarBuilder,
) -> Result<Conv3d> {
    let (kt, kh, kw) = kernel_size;
    let (st, sh, sw) = stride;
    let (_pt, ph, pw) = padding;

    // Create causal config with replicate padding
    let config = Conv3dConfig {
        kernel: (kt, kh, kw),
        stride: (st, sh, sw),
        padding: (0, ph, pw), // Temporal padding handled by causal mode
        dilation: (1, 1, 1),
        groups: 1,
        is_causal: true,
        padding_mode: PaddingMode::Replicate,
    };

    Conv3d::new(in_channels, out_channels, config, vb)
}

// =============================================================================
// AvgDown3D / DupUp3D - Downsampling and Upsampling
// =============================================================================

/// Average pooling 3D downsampler.
#[derive(Debug, Clone)]
pub struct AvgDown3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    factor: usize,
    group_size: usize,
}

impl AvgDown3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        factor_t: usize,
        factor_s: usize,
    ) -> Result<Self> {
        let factor = factor_t * factor_s * factor_s;
        if !(in_channels * factor).is_multiple_of(out_channels) {
            candle_core::bail!(
                "in_channels*factor must be divisible by out_channels: {}*{} % {} != 0",
                in_channels,
                factor,
                out_channels
            );
        }
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
        let (_, c, t, _, _) = x.dims5()?;
        if c != self.in_channels {
            candle_core::bail!(
                "AvgDown3D channel mismatch: expected {}, got {}",
                self.in_channels,
                c
            );
        }

        // Pad time if needed
        let pad_t = (self.factor_t - (t % self.factor_t)) % self.factor_t;
        let x = pad_time_front(x, pad_t)?;
        let (b, c, t2, h2, w2) = x.dims5()?;

        let ft = self.factor_t;
        let fs = self.factor_s;
        let t_out = t2 / ft;
        let h_out = h2 / fs;
        let w_out = w2 / fs;

        // Reshape for averaging - split into multiple steps to avoid 8D
        // Step 1: Reshape to [b, c, t_out, ft, h2, w2]
        let x = x.reshape(&[b, c, t_out, ft, h2, w2])?;
        // Step 2: Permute to [b, c, ft, t_out, h2, w2]
        let x = x.permute([0, 1, 3, 2, 4, 5])?;
        // Step 3: Reshape to [b, c*ft, t_out, h_out, fs, w2]
        let x = x.reshape(&[b, c * ft, t_out, h_out, fs, w2])?;
        // Step 4: Permute to [b, c*ft, fs, t_out, h_out, w2]
        let x = x.permute([0, 1, 4, 2, 3, 5])?;
        // Step 5: Reshape to [b, c*ft*fs, t_out, h_out, w_out, fs]
        let x = x.reshape(&[b, c * ft * fs, t_out, h_out, w_out, fs])?;
        // Step 6: Permute to [b, c*ft*fs, fs, t_out, h_out, w_out]
        let x = x.permute([0, 1, 5, 2, 3, 4])?;
        // Step 7: Reshape to [b, c*factor, t_out, h_out, w_out]
        let x = x.reshape(&[b, c * self.factor, t_out, h_out, w_out])?;
        // Step 8: Reshape to [b, out_channels, group_size, t_out, h_out, w_out]
        let x = x.reshape(&[b, self.out_channels, self.group_size, t_out, h_out, w_out])?;
        x.mean(2)
    }
}

/// Duplication 3D upsampler.
#[derive(Debug, Clone)]
pub struct DupUp3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    repeats: usize,
}

impl DupUp3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        factor_t: usize,
        factor_s: usize,
    ) -> Result<Self> {
        let factor = factor_t * factor_s * factor_s;
        if !(out_channels * factor).is_multiple_of(in_channels) {
            candle_core::bail!(
                "out_channels*factor must be divisible by in_channels: {}*{} % {} != 0",
                out_channels,
                factor,
                in_channels
            );
        }
        let repeats = out_channels * factor / in_channels;
        Ok(Self {
            in_channels,
            out_channels,
            factor_t,
            factor_s,
            repeats,
        })
    }

    /// Forward pass matching Python diffusers implementation exactly.
    /// Uses repeat_interleave + view + permute for memory efficiency.
    pub fn forward(&self, x: &Tensor, first_chunk: bool) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        if c != self.in_channels {
            candle_core::bail!(
                "DupUp3D channel mismatch: expected {}, got {}",
                self.in_channels,
                c
            );
        }

        let ft = self.factor_t;
        let fs = self.factor_s;

        // Step 1: repeat_interleave along channel dimension
        // Python: x = x.repeat_interleave(self.repeats, dim=1)
        let x = repeat_interleave_dim1(x, self.repeats)?;
        // Now x has shape [b, in_channels * repeats, t, h, w]
        // = [b, out_channels * factor, t, h, w]
        // = [b, out_channels * ft * fs * fs, t, h, w]

        // Step 2: Reshape to [b, out_channels, ft, fs, fs, t, h, w]
        // Python: x = x.view(b, out_channels, factor_t, factor_s, factor_s, t, h, w)
        let x = x.reshape(&[b, self.out_channels, ft, fs, fs, t, h, w])?;

        // Step 3: Permute to [b, out_channels, t, ft, h, fs, w, fs]
        // Python: x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        let x = x.permute([0, 1, 5, 2, 6, 3, 7, 4])?.contiguous()?;

        // Step 4: Reshape to [b, out_channels, t*ft, h*fs, w*fs]
        // Python: x = x.view(b, out_channels, t*ft, h*fs, w*fs)
        let mut x = x.reshape(&[b, self.out_channels, t * ft, h * fs, w * fs])?;

        // Step 5: Slice if first_chunk
        // Python: if first_chunk: x = x[:, :, factor_t - 1:, :, :]
        if first_chunk {
            let start = ft - 1;
            let t_total = x.dims5()?.2;
            let len = t_total - start;
            x = slice_time(&x, start, len)?;
        }
        Ok(x)
    }
}

/// Repeat interleave along dimension 1 (channel dimension).
/// This is more memory efficient than creating copies and concatenating.
fn repeat_interleave_dim1(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (b, c, t, h, w) = x.dims5()?;
    // Reshape to [b, c, 1, t, h, w]
    let x = x.unsqueeze(2)?;
    // Expand to [b, c, repeats, t, h, w] using broadcast
    let x = x.broadcast_as(&[b, c, repeats, t, h, w])?;
    // Reshape to [b, c * repeats, t, h, w]
    x.reshape(&[b, c * repeats, t, h, w])
}

fn pad_hw_replicate(x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
    if ph == 0 && pw == 0 {
        return Ok(x.clone());
    }
    let (b, c, t, h, w) = x.dims5()?;
    
    // Process height padding
    let x = if ph > 0 {
        let top = x.narrow(3, 0, 1)?.broadcast_as((b, c, t, ph, w))?;
        let bottom = x.narrow(3, h - 1, 1)?.broadcast_as((b, c, t, ph, w))?;
        Tensor::cat(&[&top, x, &bottom], 3)?
    } else {
        x.clone()
    };
    
    // Process width padding
    let x = if pw > 0 {
        let (b_new, c_new, t_new, h_new, _) = x.dims5()?;
        let left = x.narrow(4, 0, 1)?.broadcast_as((b_new, c_new, t_new, h_new, pw))?;
        let right = x.narrow(4, w - 1, 1)?.broadcast_as((b_new, c_new, t_new, h_new, pw))?;
        Tensor::cat(&[&left, &x, &right], 4)?
    } else {
        x
    };
    
    Ok(x)
}

// =============================================================================
// Feature Cache for Causal Inference
// =============================================================================

/// Feature cache entry for causal convolution inference.
#[derive(Clone)]
pub enum FeatCache {
    Empty,
    Rep, // Sentinel for first call
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

/// Extension trait for feature cache vectors.
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

fn build_cache_frames(x: &Tensor, cache_entry: &FeatCache) -> Result<Tensor> {
    let (_, _, t, _, _) = x.dims5()?;
    let take = CACHE_T.min(t);
    let start = t - take;
    // OPTIMIZATION: .copy() is critical to break view dependency on large input activations
    let mut cache_x = slice_time(x, start, take)?.copy()?;

    if cache_x.dims5()?.2 < CACHE_T
        && let Some(prev) = cache_entry.as_tensor()
    {
        let prev_t = prev.dims5()?.2;
        let prev_last = slice_time(prev, prev_t - 1, 1)?;
        cache_x = Tensor::cat(&[&prev_last.to_device(x.device())?, &cache_x], 2)?;
    }

    if cache_x.dims5()?.2 < CACHE_T {
        let (b2, c2, t2, h2, w2) = cache_x.dims5()?;
        let zeros = Tensor::zeros(
            (b2, c2, CACHE_T - t2, h2, w2),
            cache_x.dtype(),
            cache_x.device(),
        )?;
        cache_x = Tensor::cat(&[&zeros, &cache_x], 2)?;
    }

    Ok(cache_x)
}

// =============================================================================
// WanResample - Resampling Module
// =============================================================================

/// Resampling mode for WanResample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    None,
    Upsample2d,
    Upsample3d,
    Downsample2d,
    Downsample3d,
}

/// Resampling module for encoder/decoder.
#[derive(Debug)]
pub struct WanResample {
    mode: ResampleMode,
    conv2d: Option<Conv2d>,
    time_conv: Option<WanCausalConv3d>,
    #[allow(dead_code)]
    out_dim: usize,
}

impl WanResample {
    pub fn new(
        dim: usize,
        mode: ResampleMode,
        upsample_out_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_dim = upsample_out_dim.unwrap_or(dim / 2);

        let (conv2d, time_conv) = match mode {
            ResampleMode::None => (None, None),
            ResampleMode::Upsample2d => {
                // Conv2d with padding=1, kernel=3
                let conv = Conv2d::new(
                    vb.pp("resample.1").get((out_dim, dim, 3, 3), "weight")?,
                    Some(vb.pp("resample.1").get(out_dim, "bias")?),
                    Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                        ..Default::default()
                    },
                );
                (Some(conv), None)
            }
            ResampleMode::Upsample3d => {
                let conv = Conv2d::new(
                    vb.pp("resample.1").get((out_dim, dim, 3, 3), "weight")?,
                    Some(vb.pp("resample.1").get(out_dim, "bias")?),
                    Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                        ..Default::default()
                    },
                );
                let tc = wan_causal_conv3d_new(
                    dim,
                    dim * 2,
                    (3, 1, 1),
                    (1, 1, 1),
                    (1, 0, 0),
                    vb.pp("time_conv"),
                )?;
                (Some(conv), Some(tc))
            }
            ResampleMode::Downsample2d => {
                // ZeroPad2d(0,1,0,1) + Conv2d stride=2
                let conv = Conv2d::new(
                    vb.pp("resample.1").get((dim, dim, 3, 3), "weight")?,
                    Some(vb.pp("resample.1").get(dim, "bias")?),
                    Conv2dConfig {
                        padding: 0, // We'll handle padding manually
                        stride: 2,
                        dilation: 1,
                        groups: 1,
                        ..Default::default()
                    },
                );
                (Some(conv), None)
            }
            ResampleMode::Downsample3d => {
                let conv = Conv2d::new(
                    vb.pp("resample.1").get((dim, dim, 3, 3), "weight")?,
                    Some(vb.pp("resample.1").get(dim, "bias")?),
                    Conv2dConfig {
                        padding: 0,
                        stride: 2,
                        dilation: 1,
                        groups: 1,
                        ..Default::default()
                    },
                );
                let tc = wan_causal_conv3d_new(
                    dim,
                    dim,
                    (3, 1, 1),
                    (2, 1, 1),
                    (0, 0, 0),
                    vb.pp("time_conv"),
                )?;
                (Some(conv), Some(tc))
            }
        };

        Ok(Self {
            mode,
            conv2d,
            time_conv,
            out_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        feat_cache: Option<&mut Vec<FeatCache>>,
        feat_idx: &mut usize,
    ) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;

        match self.mode {
            // OPTIMIZATION: Return reference directly, not a clone
            ResampleMode::None => Ok(x.clone()),

            ResampleMode::Upsample2d => {
                let t = x.dims5()?.2;
                eprintln!("      Resample: Upsample2d start (T={})...", t);
                // Ensure input is contiguous to avoid hidden copies in reshape
                let x_cont = x.contiguous()?;
                
                let x_4d = if t == 1 {
                    x_cont.reshape((b, c, h, w))?
                } else {
                    x_cont.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((b * t, c, h, w))?
                };
                
                let x2 = upsample_nearest_2x(&x_4d)?;
                // Force synchronize to free temp cuDNN buffers before conv
                x2.device().synchronize()?;
                
                let x2 = x2.apply(self.conv2d.as_ref().unwrap())?;
                // Sync after heavy 480x480 conv
                x2.device().synchronize()?;
                
                let (_, c2, h2, w2) = x2.dims4()?;

                if t == 1 {
                    x2.reshape((b, c2, 1, h2, w2))
                } else {
                    x2.reshape((b, t, c2, h2, w2))?.permute((0, 2, 1, 3, 4))
                }
            }

            ResampleMode::Upsample3d => {
                eprintln!("      Resample: Upsample3d start...");
                // Time conv with cache
                let x_tc: Tensor;
                
                if let Some(cache) = feat_cache {
                    let idx = *feat_idx;
                    cache.ensure_len(idx + 1);
                    if cache[idx].is_empty() {
                        cache[idx] = FeatCache::Rep;
                        *feat_idx += 1;
                        // First chunk: no time conv, just pass through to 2D upsample
                        x_tc = x.clone();
                    } else {
                        eprintln!("      Resample: Upsample3d time_conv forward...");
                        let tc = self.time_conv.as_ref().unwrap();
                        // Use native Conv3d's forward_with_cache
                        let cache_tensor = if matches!(cache[idx], FeatCache::Rep) {
                            None
                        } else {
                            cache[idx].as_tensor()
                        };
                        let (x_conv, new_cache) = tc.forward_with_cache(x, cache_tensor)?;
                        if let Some(nc) = new_cache {
                            cache[idx] = FeatCache::Tensor(nc);
                        }
                        *feat_idx += 1;

                        eprintln!("      Resample: Upsample3d reshape/stack...");
                        // Reshape [b, 2*c, t, h, w] -> interleave time
                        let x_tc2 = x_conv.reshape((b, 2, c, t, h, w))?;
                        let a = x_tc2.i((.., 0, .., .., .., ..))?;
                        let b2 = x_tc2.i((.., 1, .., .., .., ..))?;
                        x_tc = Tensor::stack(&[&a, &b2], 3)?.reshape((b, c, t * 2, h, w))?;
                    }
                } else {
                    eprintln!("      Resample: Upsample3d (no cache) time_conv forward...");
                    let tc = self.time_conv.as_ref().unwrap();
                    let x_conv = tc.forward(x)?;
                    let x_tc2 = x_conv.reshape((b, 2, c, t, h, w))?;
                    let a = x_tc2.i((.., 0, .., .., .., ..))?;
                    let b2 = x_tc2.i((.., 1, .., .., .., ..))?;
                    x_tc = Tensor::stack(&[&a, &b2], 3)?.reshape((b, c, t * 2, h, w))?;
                }

                eprintln!("      Resample: Upsample3d upsample_nearest_2x...");
                // 2D upsample + conv
                let (b2, c2, t2, h2, w2) = x_tc.dims5()?;
                let x_4d = if t2 == 1 {
                    x_tc.reshape((b2, c2, h2, w2))?
                } else {
                    x_tc.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((b2 * t2, c2, h2, w2))?
                };
                
                // Clear x_tc early if possible
                if t2 > 1 { drop(x_tc); } 

                let x2 = upsample_nearest_2x(&x_4d)?;
                x2.device().synchronize()?;
                let x2 = x2.apply(self.conv2d.as_ref().unwrap())?;
                x2.device().synchronize()?;
                let (_, c3, h3, w3) = x2.dims4()?;
                
                if t2 == 1 {
                    x2.reshape((b2, c3, 1, h3, w3))
                } else {
                    x2.reshape((b2, t2, c3, h3, w3))?.permute((0, 2, 1, 3, 4))
                }
            }

            ResampleMode::Downsample2d => {
                // ZeroPad2d(0,1,0,1) then conv stride=2
                let x2 = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
                let x2 = zero_pad_2d(&x2, 0, 1, 0, 1)?;
                let x2 = x2.apply(self.conv2d.as_ref().unwrap())?;
                let (_, c2, h2, w2) = x2.dims4()?;
                x2.reshape((b, t, c2, h2, w2))?.permute((0, 2, 1, 3, 4))
            }

            ResampleMode::Downsample3d => {
                // OPTIMIZATION: Removed x.clone() - work with reference where possible
                if let Some(cache) = feat_cache {
                    let idx = *feat_idx;
                    cache.ensure_len(idx + 1);
                    if cache[idx].is_empty() {
                        // FIX: Store only the LAST FRAME, not the entire tensor!
                        // This matches Python: feat_cache[idx] = x.clone() but we're smarter
                        // For first chunk, we need to store last frame for next iteration
                        let x_t = x.dims5()?.2;
                        let cache_x = slice_time(x, x_t - 1, 1)?.copy()?;
                        cache[idx] = FeatCache::Tensor(cache_x);
                        *feat_idx += 1;
                        // No time_conv on first chunk - just pass through to 2D downsample
                    } else {
                        let prev_tensor = cache[idx].as_tensor().unwrap();
                        let prev_t = prev_tensor.dims5()?.2;
                        let prev_last = slice_time(prev_tensor, prev_t - 1, 1)?;
                        let x_cat = Tensor::cat(&[&prev_last, x], 2)?;
                        let tc = self.time_conv.as_ref().unwrap();
                        // Use native Conv3d forward (no cache needed here since we manually concatenate)
                        let x_conv = tc.forward(&x_cat)?;
                        
                        // Store only last frame for next iteration
                        let x_t = x.dims5()?.2;
                        let cache_x = slice_time(x, x_t - 1, 1)?.copy()?;
                        cache[idx] = FeatCache::Tensor(cache_x);
                        *feat_idx += 1;

                        // 2D downsample on conv result
                        let (b2, c2, t2, h2, w2) = x_conv.dims5()?;
                        let x2 = x_conv
                            .permute((0, 2, 1, 3, 4))?
                            .reshape((b2 * t2, c2, h2, w2))?;
                        let x2 = zero_pad_2d(&x2, 0, 1, 0, 1)?;
                        let x2 = x2.apply(self.conv2d.as_ref().unwrap())?;
                        let (_, c3, h3, w3) = x2.dims4()?;
                        return x2.reshape((b2, t2, c3, h3, w3))?.permute((0, 2, 1, 3, 4));
                    }
                } else {
                    let tc = self.time_conv.as_ref().unwrap();
                    // Use native Conv3d forward
                    let x_conv = tc.forward(x)?;
                    
                    // 2D downsample
                    let (b2, c2, t2, h2, w2) = x_conv.dims5()?;
                    let x2 = x_conv
                        .permute((0, 2, 1, 3, 4))?
                        .reshape((b2 * t2, c2, h2, w2))?;
                    let x2 = zero_pad_2d(&x2, 0, 1, 0, 1)?;
                    let x2 = x2.apply(self.conv2d.as_ref().unwrap())?;
                    let (_, c3, h3, w3) = x2.dims4()?;
                    return x2.reshape((b2, t2, c3, h3, w3))?.permute((0, 2, 1, 3, 4));
                }

                // First chunk path continues here (no time_conv)
                let (b2, c2, t2, h2, w2) = x.dims5()?;
                let x2 = x
                    .permute((0, 2, 1, 3, 4))?
                    .reshape((b2 * t2, c2, h2, w2))?;
                let x2 = zero_pad_2d(&x2, 0, 1, 0, 1)?;
                let x2 = x2.apply(self.conv2d.as_ref().unwrap())?;
                let (_, c3, h3, w3) = x2.dims4()?;
                x2.reshape((b2, t2, c3, h3, w3))?.permute((0, 2, 1, 3, 4))
            }
        }
    }
}

/// Nearest neighbor 2x upsampling using interpolation.
/// This is more memory efficient than using repeat().
fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (_b, _c, h, w) = x.dims4()?;
    // Use candle's upsample_nearest2d which is memory efficient
    x.upsample_nearest2d(h * 2, w * 2)
}

fn zero_pad_2d(x: &Tensor, left: usize, right: usize, top: usize, bottom: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let h2 = h + top + bottom;
    let w2 = w + left + right;
    let mut out = Tensor::zeros((b, c, h2, w2), x.dtype(), x.device())?;
    out = out.slice_assign(&[0..b, 0..c, top..(top + h), left..(left + w)], x)?;
    Ok(out)
}

// =============================================================================
// WanResidualBlock - Residual Block
// =============================================================================

/// Residual block for Wan VAE.
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
        let norm1 = WanRmsNorm::new(in_dim, false, vb.pp("norm1"))?;
        let conv1 = WanCausalConv3d::new(
            in_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv1"),
        )?;
        let norm2 = WanRmsNorm::new(out_dim, false, vb.pp("norm2"))?;
        let conv2 = WanCausalConv3d::new(
            out_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv2"),
        )?;

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
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mut feat_cache: Option<&mut Vec<FeatCache>>,
        feat_idx: &mut usize,
    ) -> Result<Tensor> {
        // Shortcut
        let h = if let Some(sc) = &self.conv_shortcut {
            sc.forward(x, None)?
        } else {
            x.clone()
        };

        // Main path
        let mut y = self.norm1.forward(x)?;
        y = silu(&y)?;

        // Conv1 with cache - reborrow to allow multiple uses
        y = causal_conv_cached(&self.conv1, &y, feat_cache.as_deref_mut(), feat_idx)?;

        y = self.norm2.forward(&y)?;
        y = silu(&y)?;

        // Conv2 with cache
        y = causal_conv_cached(&self.conv2, &y, feat_cache, feat_idx)?;

        y.add(&h)
    }
}

fn causal_conv_cached(
    conv: &Conv3d,
    x: &Tensor,
    feat_cache: Option<&mut Vec<FeatCache>>,
    feat_idx: &mut usize,
) -> Result<Tensor> {
    if let Some(cache) = feat_cache {
        let idx = *feat_idx;
        cache.ensure_len(idx + 1);
        
        // Get the existing cache tensor if available
        let cache_tensor = cache[idx].as_tensor();
        
        // Use native Conv3d's forward_with_cache
        let (out, new_cache) = conv.forward_with_cache(x, cache_tensor)?;
        
        // Store the new cache
        if let Some(nc) = new_cache {
            cache[idx] = FeatCache::Tensor(nc);
        }
        *feat_idx += 1;
        Ok(out)
    } else {
        // No caching - use regular forward
        conv.forward(x)
    }
}

// =============================================================================
// WanAttentionBlock - Self-Attention Block
// =============================================================================

/// Single-head self-attention block for Wan VAE.
#[derive(Debug)]
pub struct WanAttentionBlock {
    norm: WanRmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
    #[allow(dead_code)]
    dim: usize,
}

impl WanAttentionBlock {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = WanRmsNorm::new(dim, true, vb.pp("norm"))?;
        let to_qkv = Conv2d::new(
            vb.pp("to_qkv").get((dim * 3, dim, 1, 1), "weight")?,
            Some(vb.pp("to_qkv").get(dim * 3, "bias")?),
            Conv2dConfig::default(),
        );
        let proj = Conv2d::new(
            vb.pp("proj").get((dim, dim, 1, 1), "weight")?,
            Some(vb.pp("proj").get(dim, "bias")?),
            Conv2dConfig::default(),
        );

        Ok(Self {
            norm,
            to_qkv,
            proj,
            dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let (b, c, t, h, w) = x.dims5()?;

        // Reshape to [b*t, c, h, w]
        let mut y = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
        y = self.norm.forward(&y)?;

        // QKV projection
        let qkv = y.apply(&self.to_qkv)?; // [b*t, 3c, h, w]
        let hw = h * w;
        let qkv = qkv.reshape((b * t, 1, 3 * c, hw))?.permute((0, 1, 3, 2))?; // [bt, 1, hw, 3c]

        // Split QKV using narrow
        let q = qkv.narrow(3, 0, c)?;
        let k = qkv.narrow(3, c, c)?;
        let v = qkv.narrow(3, 2 * c, c)?;

        // Scaled dot-product attention
        let scale = 1f32 / (c as f32).sqrt();
        let k_t = k.transpose(2, 3)?;
        let scores = (q.matmul(&k_t)? * scale as f64)?;
        let probs = candle_nn::ops::softmax(&scores, 3)?;
        let y = probs.matmul(&v)?; // [bt, 1, hw, c]

        let y = y
            .squeeze(1)?
            .permute((0, 2, 1))?
            .reshape((b * t, c, h, w))?;
        let y = y.apply(&self.proj)?;

        // Reshape back to [b, c, t, h, w]
        let y = y.reshape((b, t, c, h, w))?.permute((0, 2, 1, 3, 4))?;

        y.add(&identity)
    }
}

// =============================================================================
// WanMidBlock - Middle Block
// =============================================================================

/// Middle block with residual blocks and attention.
#[derive(Debug)]
pub struct WanMidBlock {
    resnets: Vec<WanResidualBlock>,
    attentions: Vec<WanAttentionBlock>,
}

impl WanMidBlock {
    pub fn new(dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers + 1);
        let mut attentions = Vec::with_capacity(num_layers);

        resnets.push(WanResidualBlock::new(dim, dim, vb.pp("resnets.0"))?);

        for i in 0..num_layers {
            attentions.push(WanAttentionBlock::new(
                dim,
                vb.pp(format!("attentions.{}", i)),
            )?);
            resnets.push(WanResidualBlock::new(
                dim,
                dim,
                vb.pp(format!("resnets.{}", i + 1)),
            )?);
        }

        Ok(Self {
            resnets,
            attentions,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mut feat_cache: Option<&mut Vec<FeatCache>>,
        feat_idx: &mut usize,
    ) -> Result<Tensor> {
        let mut y = self.resnets[0].forward(x, feat_cache.as_deref_mut(), feat_idx)?;

        for (attn, resnet) in self.attentions.iter().zip(self.resnets.iter().skip(1)) {
            y = attn.forward(&y)?;
            y = resnet.forward(&y, feat_cache.as_deref_mut(), feat_idx)?;
        }

        Ok(y)
    }
}

// =============================================================================
// WanUpBlock - Upsampling Block (Wan 2.1)
// =============================================================================

/// Upsampling block for Wan 2.1 decoder.
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
        let mut current_dim = in_dim;

        for i in 0..(num_res_blocks + 1) {
            resnets.push(WanResidualBlock::new(
                current_dim,
                out_dim,
                vb.pp(format!("resnets.{}", i)),
            )?);
            current_dim = out_dim;
        }

        let upsampler = if let Some(mode) = upsample_mode {
            Some(WanResample::new(
                out_dim,
                mode,
                None, // Use default out_dim = dim / 2
                vb.pp("upsamplers.0"),
            )?)
        } else {
            None
        };

        Ok(Self { resnets, upsampler })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mut feat_cache: Option<&mut Vec<FeatCache>>,
        feat_idx: &mut usize,
        _first_chunk: bool,
    ) -> Result<Tensor> {
        // OPTIMIZATION: Start with shallow clone, not deep clone
        let mut y = x.clone();

        for (i, resnet) in self.resnets.iter().enumerate() {
            eprintln!("    up_block resnet {}...", i);
            let next_y = resnet.forward(&y, feat_cache.as_deref_mut(), feat_idx)?;
            // Free old y explicitly to reuse memory
            drop(y);
            y = next_y;
            eprintln!("    up_block resnet {} done, shape {:?}", i, y.dims());
        }

        if let Some(ups) = &self.upsampler {
            eprintln!("    up_block upsampler...");
            let next_y = ups.forward(&y, feat_cache, feat_idx)?;
            drop(y);
            y = next_y;
            eprintln!("    up_block upsampler done, shape {:?}", y.dims());
        }

        Ok(y)
    }
}

// =============================================================================
// WanEncoder3d - 3D Encoder
// =============================================================================

/// 3D Encoder for Wan VAE.
#[derive(Debug)]
pub struct WanEncoder3d {
    conv_in: WanCausalConv3d,
    down_blocks: Vec<WanEncoderBlock>,
    mid_block: WanMidBlock,
    norm_out: WanRmsNorm,
    conv_out: WanCausalConv3d,
}

/// Encoder block (residual blocks + optional downsample).
#[derive(Debug)]
pub enum WanEncoderBlock {
    Residual(WanResidualBlock),
    Attention(WanAttentionBlock),
    Resample(WanResample),
}

impl WanEncoder3d {
    pub fn new(cfg: &AutoencoderKLWanConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.base_dim;
        let z_dim = cfg.z_dim;
        let dim_mult = &cfg.dim_mult;
        let num_res_blocks = cfg.num_res_blocks;
        let temporal_downsample = &cfg.temporal_downsample;

        // Dimensions: [dim, dim*1, dim*2, dim*4, dim*4]
        let mut dims = vec![dim];
        for &m in dim_mult {
            dims.push(dim * m);
        }

        // conv_in
        let conv_in = WanCausalConv3d::new(
            cfg.in_channels,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_in"),
        )?;

        // down_blocks
        let mut down_blocks = Vec::new();
        let mut block_idx = 0;

        for i in 0..dim_mult.len() {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];

            // Residual blocks
            let mut current_in = in_dim;
            for _ in 0..num_res_blocks {
                down_blocks.push(WanEncoderBlock::Residual(WanResidualBlock::new(
                    current_in,
                    out_dim,
                    vb.pp(format!("down_blocks.{}", block_idx)),
                )?));
                current_in = out_dim;
                block_idx += 1;
            }

            // Downsample (except last stage)
            if i != dim_mult.len() - 1 {
                let mode = if temporal_downsample[i] {
                    ResampleMode::Downsample3d
                } else {
                    ResampleMode::Downsample2d
                };
                down_blocks.push(WanEncoderBlock::Resample(WanResample::new(
                    out_dim,
                    mode,
                    None,
                    vb.pp(format!("down_blocks.{}", block_idx)),
                )?));
                block_idx += 1;
            }
        }

        // mid_block
        let final_dim = dims[dim_mult.len()];
        let mid_block = WanMidBlock::new(final_dim, 1, vb.pp("mid_block"))?;

        // norm_out + conv_out
        let norm_out = WanRmsNorm::new(final_dim, false, vb.pp("norm_out"))?;
        let conv_out = WanCausalConv3d::new(
            final_dim,
            z_dim * 2, // Output 2*z_dim for mean and logvar
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_out,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mut feat_cache: Option<&mut Vec<FeatCache>>,
        feat_idx: &mut usize,
    ) -> Result<Tensor> {
        // conv_in with cache
        let mut y = causal_conv_cached(&self.conv_in, x, feat_cache.as_deref_mut(), feat_idx)?;

        // down_blocks
        for block in &self.down_blocks {
            y = match block {
                WanEncoderBlock::Residual(r) => {
                    r.forward(&y, feat_cache.as_deref_mut(), feat_idx)?
                }
                WanEncoderBlock::Attention(a) => a.forward(&y)?,
                WanEncoderBlock::Resample(s) => {
                    s.forward(&y, feat_cache.as_deref_mut(), feat_idx)?
                }
            };
        }

        // mid_block
        y = self
            .mid_block
            .forward(&y, feat_cache.as_deref_mut(), feat_idx)?;

        // head
        y = self.norm_out.forward(&y)?;
        y = silu(&y)?;
        y = causal_conv_cached(&self.conv_out, &y, feat_cache, feat_idx)?;

        Ok(y)
    }
}

// =============================================================================
// WanDecoder3d - 3D Decoder
// =============================================================================

/// 3D Decoder for Wan VAE.
#[derive(Debug)]
pub struct WanDecoder3d {
    conv_in: WanCausalConv3d,
    mid_block: WanMidBlock,
    up_blocks: Vec<WanUpBlock>,
    norm_out: WanRmsNorm,
    conv_out: WanCausalConv3d,
}

impl WanDecoder3d {
    pub fn new(cfg: &AutoencoderKLWanConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.get_decoder_base_dim();
        let z_dim = cfg.z_dim;
        let dim_mult = &cfg.dim_mult;
        let num_res_blocks = cfg.num_res_blocks;
        let temporal_upsample: Vec<bool> = cfg.temporal_downsample.iter().rev().cloned().collect();

        // Dimensions reversed: [dim*4, dim*4, dim*2, dim*1, dim]
        let mut dims = vec![dim * dim_mult[dim_mult.len() - 1]];
        for &m in dim_mult.iter().rev() {
            dims.push(dim * m);
        }

        // conv_in
        let conv_in = WanCausalConv3d::new(
            z_dim,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_in"),
        )?;

        // mid_block
        let mid_block = WanMidBlock::new(dims[0], 1, vb.pp("mid_block"))?;

        // up_blocks
        let mut up_blocks = Vec::with_capacity(dim_mult.len());
        for i in 0..dim_mult.len() {
            let in_dim = if i > 0 { dims[i] / 2 } else { dims[i] }; // Wan 2.1 halves input
            let out_dim = dims[i + 1];

            let up_flag = i != dim_mult.len() - 1;
            let upsample_mode = if up_flag {
                if temporal_upsample[i] {
                    Some(ResampleMode::Upsample3d)
                } else {
                    Some(ResampleMode::Upsample2d)
                }
            } else {
                None
            };

            up_blocks.push(WanUpBlock::new(
                in_dim,
                out_dim,
                num_res_blocks,
                upsample_mode,
                vb.pp(format!("up_blocks.{}", i)),
            )?);
        }

        // norm_out + conv_out
        let final_dim = dims[dim_mult.len()];
        let norm_out = WanRmsNorm::new(final_dim, false, vb.pp("norm_out"))?;
        let conv_out = WanCausalConv3d::new(
            final_dim,
            cfg.out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mut feat_cache: Option<&mut Vec<FeatCache>>,
        feat_idx: &mut usize,
        first_chunk: bool,
    ) -> Result<Tensor> {
        eprintln!("  DEBUG decoder: conv_in...");
        // conv_in with cache
        let mut y = causal_conv_cached(&self.conv_in, x, feat_cache.as_deref_mut(), feat_idx)?;
        eprintln!("  DEBUG decoder: conv_in done, shape {:?}", y.dims());

        eprintln!("  DEBUG decoder: mid_block...");
        // mid_block
        y = self
            .mid_block
            .forward(&y, feat_cache.as_deref_mut(), feat_idx)?;
        eprintln!("  DEBUG decoder: mid_block done, shape {:?}", y.dims());

        // up_blocks
        for (i, up_block) in self.up_blocks.iter().enumerate() {
        eprintln!("  DEBUG decoder: up_block {}...", i);
        let next_y = up_block.forward(&y, feat_cache.as_deref_mut(), feat_idx, first_chunk)?;
        drop(y);
        y = next_y;
        // Force sync after each heavy block
        y.device().synchronize()?;
        eprintln!("  DEBUG decoder: up_block {} done, shape {:?}", i, y.dims());
        }

        eprintln!("  DEBUG decoder: head...");
        // head
        y = self.norm_out.forward(&y)?;
        y = silu(&y)?;
        y = causal_conv_cached(&self.conv_out, &y, feat_cache, feat_idx)?;
        eprintln!("  DEBUG decoder: head done, shape {:?}", y.dims());

        Ok(y)
    }
}

// =============================================================================
// AutoencoderKLWan - Main VAE Model
// =============================================================================

/// AutoencoderKLWan - 3D VAE for Wan video generation.
#[derive(Debug)]
pub struct AutoencoderKLWan {
    pub cfg: AutoencoderKLWanConfig,
    pub encoder: WanEncoder3d,
    pub quant_conv: WanCausalConv3d,
    pub post_quant_conv: WanCausalConv3d,
    pub decoder: WanDecoder3d,

    // Tiling settings
    pub use_slicing: bool,
    pub use_tiling: bool,
    pub tile_sample_min_height: usize,
    pub tile_sample_min_width: usize,
    pub tile_sample_stride_height: usize,
    pub tile_sample_stride_width: usize,

    // Cache counts (precomputed)
    decoder_conv_num: usize,
    encoder_conv_num: usize,
}

impl AutoencoderKLWan {
    /// Create a new AutoencoderKLWan from config and VarBuilder.
    pub fn new(cfg: AutoencoderKLWanConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = WanEncoder3d::new(&cfg, vb.pp("encoder"))?;
        let quant_conv = WanCausalConv3d::new(
            cfg.z_dim * 2,
            cfg.z_dim * 2,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            vb.pp("quant_conv"),
        )?;
        let post_quant_conv = WanCausalConv3d::new(
            cfg.z_dim,
            cfg.z_dim,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            vb.pp("post_quant_conv"),
        )?;
        let decoder = WanDecoder3d::new(&cfg, vb.pp("decoder"))?;

        // Count convolutions for cache sizing
        // This is approximate - actual count depends on architecture
        let encoder_conv_num = 50; // Approximate
        let decoder_conv_num = 50; // Approximate

        Ok(Self {
            cfg,
            encoder,
            quant_conv,
            post_quant_conv,
            decoder,
            use_slicing: false,
            use_tiling: false,
            tile_sample_min_height: 256,
            tile_sample_min_width: 256,
            tile_sample_stride_height: 192,
            tile_sample_stride_width: 192,
            decoder_conv_num,
            encoder_conv_num,
        })
    }

    /// Get latent dimension.
    pub fn z_dim(&self) -> usize {
        self.cfg.z_dim
    }

    /// Get latent mean for normalization.
    pub fn latents_mean(&self) -> &[f32] {
        &self.cfg.latents_mean
    }

    /// Get latent std for normalization.
    pub fn latents_std(&self) -> &[f32] {
        &self.cfg.latents_std
    }

    /// Get spatial compression factor.
    pub fn scale_factor_spatial(&self) -> usize {
        self.cfg.scale_factor_spatial
    }

    /// Get temporal compression factor.
    pub fn scale_factor_temporal(&self) -> usize {
        self.cfg.scale_factor_temporal
    }

    /// Enable tiling for large videos.
    pub fn enable_tiling(
        &mut self,
        tile_sample_min_height: Option<usize>,
        tile_sample_min_width: Option<usize>,
        tile_sample_stride_height: Option<usize>,
        tile_sample_stride_width: Option<usize>,
    ) {
        self.use_tiling = true;
        if let Some(v) = tile_sample_min_height {
            self.tile_sample_min_height = v;
        }
        if let Some(v) = tile_sample_min_width {
            self.tile_sample_min_width = v;
        }
        if let Some(v) = tile_sample_stride_height {
            self.tile_sample_stride_height = v;
        }
        if let Some(v) = tile_sample_stride_width {
            self.tile_sample_stride_width = v;
        }
    }

    /// Disable tiling.
    pub fn disable_tiling(&mut self) {
        self.use_tiling = false;
    }

    /// Enable slicing for batch processing.
    pub fn enable_slicing(&mut self) {
        self.use_slicing = true;
    }

    /// Disable slicing.
    pub fn disable_slicing(&mut self) {
        self.use_slicing = false;
    }

    /// Encode video to latent distribution.
    pub fn encode(&self, x: &Tensor) -> Result<DiagonalGaussianDistribution> {
        let h = if self.use_slicing && x.dims5()?.0 > 1 {
            // Process batch slices
            let b = x.dims5()?.0;
            let mut slices = Vec::with_capacity(b);
            for i in 0..b {
                let xs = x.narrow(0, i, 1)?;
                slices.push(self.encode_impl(&xs)?);
            }
            Tensor::cat(&slices.iter().collect::<Vec<_>>(), 0)?
        } else {
            self.encode_impl(x)?
        };

        DiagonalGaussianDistribution::new(&h)
    }

    fn encode_impl(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, num_frames, _height, _width) = x.dims5()?;

        // Initialize cache
        let mut enc_feat_map: Vec<FeatCache> = vec![FeatCache::Empty; self.encoder_conv_num];

        // Process in temporal chunks
        // OPTIMIZATION: Collect all chunks first, then do a single concat at the end
        let iter = 1 + (num_frames - 1) / 4;
        let mut encoded_chunks: Vec<Tensor> = Vec::with_capacity(iter);

        for i in 0..iter {
            let mut enc_conv_idx = 0;
            let chunk = if i == 0 {
                slice_time(x, 0, 1)?
            } else {
                let start = 1 + 4 * (i - 1);
                let end = (1 + 4 * i).min(num_frames);
                slice_time(x, start, end - start)?
            };

            let chunk_out =
                self.encoder
                    .forward(&chunk, Some(&mut enc_feat_map), &mut enc_conv_idx)?;
            encoded_chunks.push(chunk_out);
        }

        // Clear cache
        drop(enc_feat_map);

        // Single concat at the end - O(n) instead of O(n²)
        let out = if encoded_chunks.len() == 1 {
            encoded_chunks.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Tensor> = encoded_chunks.iter().collect();
            Tensor::cat(&refs, 2)?
        };

        self.quant_conv.forward(&out, None)
    }

    /// Decode latent to video.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let decoded = if self.use_slicing && z.dims5()?.0 > 1 {
            let b = z.dims5()?.0;
            let mut slices = Vec::with_capacity(b);
            for i in 0..b {
                let zs = z.narrow(0, i, 1)?;
                slices.push(self.decode_impl(&zs)?);
            }
            Tensor::cat(&slices.iter().collect::<Vec<_>>(), 0)?
        } else {
            self.decode_impl(z)?
        };

        Ok(decoded)
    }

    fn decode_impl(&self, z: &Tensor) -> Result<Tensor> {
        let (_, _, num_frames, _height, _width) = z.dims5()?;

        // Initialize cache
        let mut feat_map: Vec<FeatCache> = vec![FeatCache::Empty; self.decoder_conv_num];

        // Post-quant conv - processes all frames at once (1x1x1 conv, low memory)
        let x = self.post_quant_conv.forward(z, None)?;

        // Process frame by frame to save memory (matching Python diffusers exactly)
        // OPTIMIZATION: Collect all frames first, then do a single concat at the end
        // This changes O(n²) memory complexity to O(n) by avoiding growing tensors
        // 
        // NOTE: Memory usage here is dominated by:
        // 1. feat_map cache (~50 tensors with 2 frames each at various resolutions)
        // 2. decoded_frames vector (all decoded outputs before concat)
        // For 480x480 at 33 frames, this can peak at ~8-10GB on GPU
        let mut decoded_frames: Vec<Tensor> = Vec::with_capacity(num_frames);
        
        eprintln!("DEBUG: Starting decode loop for {} frames", num_frames);
        
        for i in 0..num_frames {
            eprintln!("DEBUG: Decoding frame {}/{}", i + 1, num_frames);
            let mut conv_idx = 0;
            let frame = slice_time(&x, i, 1)?;
            let decoded =
                self.decoder
                    .forward(&frame, Some(&mut feat_map), &mut conv_idx, i == 0)?;
            
            // OPTIMIZATION: Ensure decoded is a clean copy to release intermediate VRAM
            decoded_frames.push(decoded.copy()?);
            eprintln!("DEBUG: Frame {} complete", i + 1);
            
            // Explicitly sync to driver every frame
            frame.device().synchronize()?;
        }

        // Clear cache to free memory BEFORE concat
        for entry in feat_map.iter_mut() {
            *entry = FeatCache::Empty;
        }
        drop(feat_map);

        // Single concat at the end - O(n) instead of O(n²)
        let out = if decoded_frames.len() == 1 {
            decoded_frames.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Tensor> = decoded_frames.iter().collect();
            Tensor::cat(&refs, 2)?
        };

        // Clamp output to [-1, 1]
        let out = out.clamp(-1.0f64, 1.0f64)?;

        Ok(out)
    }

    /// Normalize latents using mean and std.
    pub fn normalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let device = latents.device();
        let dtype = latents.dtype();

        let mean = Tensor::from_vec(
            self.cfg.latents_mean.clone(),
            (1, self.cfg.z_dim, 1, 1, 1),
            device,
        )?
        .to_dtype(dtype)?;
        let std = Tensor::from_vec(
            self.cfg.latents_std.clone(),
            (1, self.cfg.z_dim, 1, 1, 1),
            device,
        )?
        .to_dtype(dtype)?;

        latents.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    /// Denormalize latents using mean and std.
    pub fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let device = latents.device();
        let dtype = latents.dtype();

        let mean = Tensor::from_vec(
            self.cfg.latents_mean.clone(),
            (1, self.cfg.z_dim, 1, 1, 1),
            device,
        )?
        .to_dtype(dtype)?;
        let std = Tensor::from_vec(
            self.cfg.latents_std.clone(),
            (1, self.cfg.z_dim, 1, 1, 1),
            device,
        )?
        .to_dtype(dtype)?;

        latents.broadcast_mul(&std)?.broadcast_add(&mean)
    }
}

// =============================================================================
// Output Types
// =============================================================================

/// Decoder output.
#[derive(Debug)]
pub struct DecoderOutput {
    pub sample: Tensor,
}

/// Encoder output.
#[derive(Debug)]
pub struct EncoderOutput {
    pub latent_dist: DiagonalGaussianDistribution,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_patchify_unpatchify() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 3, 4, 16, 16), &device).unwrap();
        let patch_size = 2;

        let patched = patchify(&x, patch_size).unwrap();
        assert_eq!(patched.dims(), &[1, 12, 4, 8, 8]);

        let unpatched = unpatchify(&patched, patch_size).unwrap();
        assert_eq!(unpatched.dims(), &[1, 3, 4, 16, 16]);
    }

    #[test]
    fn test_avg_down_3d() {
        let down = AvgDown3D::new(16, 32, 2, 2).unwrap();
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 16, 4, 8, 8), &device).unwrap();
        let y = down.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 32, 2, 4, 4]);
    }

    #[test]
    fn test_dup_up_3d() {
        let up = DupUp3D::new(32, 16, 2, 2).unwrap();
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 32, 2, 4, 4), &device).unwrap();
        let y = up.forward(&x, false).unwrap();
        assert_eq!(y.dims(), &[1, 16, 4, 8, 8]);
    }
}
