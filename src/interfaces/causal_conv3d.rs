//! Causal 3D Convolution implementation.
//!
//! Provides CausalConv3d that implements temporal causal convolution
//! via summing Conv2d slices. Used in LTX-Video and Wan VAE.

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

/// Configuration for CausalConv3d
#[derive(Clone, Copy, Debug)]
pub struct CausalConv3dConfig {
    pub stride: (usize, usize, usize),    // (t, h, w)
    pub dilation: (usize, usize, usize),  // (t, h, w)
    pub groups: usize,
    pub is_causal: bool,
}

impl Default for CausalConv3dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1, 1),
            dilation: (1, 1, 1),
            groups: 1,
            is_causal: true,
        }
    }
}

/// Causal 3D Convolution implemented via Conv2d slices.
///
/// Implements temporal causal convolution by:
/// 1. Padding temporally (replicate first/last frame)
/// 2. Applying Conv2d for each temporal kernel position
/// 3. Summing results across temporal dimension
#[derive(Debug, Clone)]
pub struct CausalConv3d {
    kernel_t: usize,
    kernel_h: usize,
    kernel_w: usize,
    config: CausalConv3dConfig,
    conv2d_slices: Vec<Conv2d>,
    bias: Option<Tensor>,
}

impl CausalConv3d {
    /// Create a new CausalConv3d layer.
    ///
    /// Weight path should point to a module with `weight` and optionally `bias`.
    /// Weight shape: (out_channels, in_channels/groups, kt, kh, kw)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),  // (t, h, w)
        config: CausalConv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let groups = config.groups;

        // Load 3D weight: (out, in/groups, kt, kh, kw)
        let w = vb.get((out_channels, in_channels / groups, kt, kh, kw), "weight")?;
        
        // Bias is optional
        let bias = vb.get(out_channels, "bias").ok();

        let hpad = kh / 2;

        // Create Conv2d for each temporal slice
        let mut conv2d_slices = Vec::with_capacity(kt);
        for ti in 0..kt {
            let w2 = w.i((.., .., ti, .., ..))?.contiguous()?;
            let c2cfg = Conv2dConfig {
                padding: hpad,
                stride: config.stride.1,
                dilation: config.dilation.1,
                groups,
                ..Default::default()
            };
            conv2d_slices.push(Conv2d::new(w2, None, c2cfg));
        }

        Ok(Self {
            kernel_t: kt,
            kernel_h: kh,
            kernel_w: kw,
            config,
            conv2d_slices,
            bias,
        })
    }

    /// Create with inner `conv` submodule (diffusers convention).
    pub fn new_with_conv_submodule(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        config: CausalConv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(in_channels, out_channels, kernel, config, vb.pp("conv"))
    }

    fn cat_tensors(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        let refs: Vec<&Tensor> = tensors.iter().collect();
        Tensor::cat(&refs, dim)
    }

    fn pad_time_replicate(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, t, _, _) = x.dims5()?;
        let kt = self.kernel_t;

        if kt <= 1 {
            return Ok(x.clone());
        }

        if self.config.is_causal {
            // Causal: pad left only
            let left = kt - 1;
            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let pad_left = first.repeat((1, 1, left, 1, 1))?;
            Self::cat_tensors(&[pad_left, x.clone()], 2)
        } else {
            // Non-causal: pad both sides
            let left = (kt - 1) / 2;
            let right = (kt - 1) / 2;

            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let last = x.i((.., .., t - 1, .., ..))?.unsqueeze(2)?;

            let pad_left = if left == 0 {
                None
            } else {
                Some(first.repeat((1, 1, left, 1, 1))?)
            };
            let pad_right = if right == 0 {
                None
            } else {
                Some(last.repeat((1, 1, right, 1, 1))?)
            };

            match (pad_left, pad_right) {
                (None, None) => Ok(x.clone()),
                (Some(pl), None) => Self::cat_tensors(&[pl, x.clone()], 2),
                (None, Some(pr)) => Self::cat_tensors(&[x.clone(), pr], 2),
                (Some(pl), Some(pr)) => Self::cat_tensors(&[pl, x.clone(), pr], 2),
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pad_time_replicate(x)?;
        let (_b, _c, t_pad, _h, _w) = x.dims5()?;

        let kt = self.kernel_t;
        let dt = self.config.dilation.0;
        let st = self.config.stride.0;

        let needed = (kt - 1) * dt + 1;
        if t_pad < needed {
            candle_core::bail!(
                "time dim too small after padding: t_pad={}, needed={}",
                t_pad,
                needed
            );
        }
        let t_out = (t_pad - needed) / st + 1;

        let mut ys: Vec<Tensor> = Vec::with_capacity(t_out);

        for to in 0..t_out {
            let base_t = to * st;

            let mut acc: Option<Tensor> = None;
            for ki in 0..kt {
                let ti = base_t + ki * dt;
                let xt = x.i((.., .., ti, .., ..))?; // (B,C,H,W)
                let yt = xt.apply(&self.conv2d_slices[ki])?; // (B,Out,H',W')
                acc = Some(match acc {
                    None => yt,
                    Some(prev) => prev.add(&yt)?,
                });
            }

            let yt = acc.expect("kt>=1 so acc is Some");
            ys.push(yt.unsqueeze(2)?); // (B,Out,1,H',W')
        }

        let y = Self::cat_tensors(&ys, 2)?; // (B,Out,T_out,H',W')

        if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, bias.dims1()?, 1, 1, 1))?;
            y.broadcast_add(&bias)
        } else {
            Ok(y)
        }
    }

    pub fn kernel_size(&self) -> (usize, usize, usize) {
        (self.kernel_t, self.kernel_h, self.kernel_w)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_causal_conv3d_config() {
        let config = CausalConv3dConfig::default();
        assert!(config.is_causal);
        assert_eq!(config.stride, (1, 1, 1));
    }
}
