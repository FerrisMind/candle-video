//! Wan 2D/3D resampling blocks (`WanResample`).

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use super::causal_conv3d::{CACHE_T, FeatCache, FeatSlot, WanCausalConv3d};

#[derive(Debug, Clone, Copy)]
pub enum ResampleMode {
    Upsample2d,
    Upsample3d,
}

#[derive(Debug)]
pub struct WanResample {
    mode: ResampleMode,
    spatial_conv: Conv2d,
    time_conv: Option<WanCausalConv3d>,
}

impl WanResample {
    pub fn new(dim: usize, mode: ResampleMode, vb: VarBuilder) -> Result<Self> {
        let out_dim = dim / 2;
        let spatial_conv = candle_nn::conv2d(
            dim,
            out_dim,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("resample").pp("1"),
        )?;
        let time_conv = match mode {
            ResampleMode::Upsample2d => None,
            ResampleMode::Upsample3d => Some(WanCausalConv3d::new(
                dim,
                dim * 2,
                (3, 1, 1),
                (1, 1, 1),
                (1, 0, 0),
                vb.pp("time_conv"),
            )?),
        };
        Ok(Self {
            mode,
            spatial_conv,
            time_conv,
        })
    }

    pub fn count_causal_conv3d(&self) -> usize {
        usize::from(self.time_conv.is_some())
    }

    fn spatial_upsample(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let x = x.permute((0, 2, 1, 3, 4))?.reshape((b * t, c, h, w))?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        let x = self.spatial_conv.forward(&x)?;
        x.reshape((b, t, x.dim(1)?, h * 2, w * 2))?
            .permute((0, 2, 1, 3, 4))
    }

    fn upsample3d_time(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let time_conv = self
            .time_conv
            .as_ref()
            .expect("upsample3d requires time_conv");
        let (b, c, t, h, w) = x.dims5()?;
        let idx = cache.next_slot();
        match cache.get(idx).clone() {
            FeatSlot::Empty => {
                cache.set(idx, FeatSlot::Rep);
                Ok(x.clone())
            }
            slot => {
                let mut cache_x = x.i((.., .., t.saturating_sub(CACHE_T).., .., ..))?;
                if cache_x.dim(2)? < CACHE_T {
                    match &slot {
                        FeatSlot::Tensor(prev) => {
                            let last = prev.i((.., .., prev.dim(2)? - 1.., .., ..))?;
                            cache_x = Tensor::cat(&[&last, &cache_x], 2)?;
                        }
                        FeatSlot::Rep => {
                            let zeros = cache_x.zeros_like()?;
                            cache_x = Tensor::cat(&[&zeros, &cache_x], 2)?;
                        }
                        FeatSlot::Empty => {}
                    }
                }
                let prev_cache = match &slot {
                    FeatSlot::Rep => None,
                    FeatSlot::Tensor(t) => Some(t.clone()),
                    FeatSlot::Empty => None,
                };
                let y = time_conv.forward(x, prev_cache.as_ref())?;
                cache.set(idx, FeatSlot::Tensor(cache_x));
                let y = y.reshape((b, 2, c, t, h, w))?;
                let a = y.i((.., 0, .., .., .., ..))?;
                let bch = y.i((.., 1, .., .., .., ..))?;
                let stacked = Tensor::stack(&[&a, &bch], 3)?;
                stacked.reshape((b, c, t * 2, h, w))
            }
        }
    }

    pub fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let mut x = x.clone();
        if matches!(self.mode, ResampleMode::Upsample3d) {
            x = self.upsample3d_time(&x, cache)?;
        }
        self.spatial_upsample(&x)
    }
}
