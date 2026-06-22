//! 3D patch embedding via Conv2d when `patch_size[0] == 1`.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

/// `patch_embedding` Conv3d with `kernel=stride=patch_size` (T2V 1.3B: `p_t=1`).
#[derive(Debug)]
pub struct WanPatchEmbedding {
    conv2d: Conv2d,
    patch_size: [usize; 3],
}

impl WanPatchEmbedding {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        patch_size: [usize; 3],
        vb: VarBuilder,
    ) -> Result<Self> {
        let [_p_t, p_h, p_w] = patch_size;
        if patch_size[0] != 1 {
            candle_core::bail!("WanPatchEmbedding currently supports patch_size[0] == 1 only");
        }
        let weight = vb
            .get((out_channels, in_channels, 1, p_h, p_w), "weight")?
            .squeeze(2)?;
        let bias = vb.get(out_channels, "bias")?;
        let cfg = Conv2dConfig {
            stride: p_h,
            padding: 0,
            ..Default::default()
        };
        let conv2d = Conv2d::new(weight, Some(bias), cfg);
        Ok(Self { conv2d, patch_size })
    }

    /// `[B, C, F, H, W]` → `[B, seq, inner_dim]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = xs.dims5()?;
        let [_p_t, p_h, p_w] = self.patch_size;
        if h % p_h != 0 || w % p_w != 0 {
            candle_core::bail!("spatial dims {h}x{w} not divisible by patch {p_h}x{p_w}");
        }
        let xs = xs.permute((0, 2, 1, 3, 4))?.reshape((b * f, c, h, w))?;
        let xs = self.conv2d.forward(&xs)?;
        let (_bf, out_c, hh, ww) = xs.dims4()?;
        let xs = xs.permute((0, 2, 3, 1))?;
        xs.reshape((b, f * hh * ww, out_c))
    }
}
