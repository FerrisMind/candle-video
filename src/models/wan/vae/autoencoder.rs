//! `AutoencoderKLWan` decode-only wrapper.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::ltx_video::loader::{LoaderError, WeightLoader};
use crate::models::wan::configs::WanVaeConfig;
use crate::models::wan::loader::{WanLayout, discover_safetensors, load_vae_var_builder};

use super::causal_conv3d::{FeatCache, WanCausalConv3d};
use super::decoder::WanDecoder3d;

const SPATIAL_COMPRESSION: usize = 8;

/// Wan 2.1 VAE (decode path only).
pub struct AutoencoderKLWan {
    config: WanVaeConfig,
    post_quant_conv: WanCausalConv3d,
    decoder: WanDecoder3d,
    latents_mean: Tensor,
    latents_std: Tensor,
    feat_cache: FeatCache,
    use_tiling: bool,
    tile_sample_min_height: usize,
    tile_sample_min_width: usize,
    tile_sample_stride_height: usize,
    tile_sample_stride_width: usize,
}

impl AutoencoderKLWan {
    pub fn new(
        config: WanVaeConfig,
        post_quant_conv: WanCausalConv3d,
        decoder: WanDecoder3d,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let z_dim = config.z_dim;
        let mean = Tensor::from_vec(config.latents_mean.clone(), (1, z_dim, 1, 1, 1), &device)?
            .to_dtype(dtype)?;
        let std = Tensor::from_vec(config.latents_std.clone(), (1, z_dim, 1, 1, 1), &device)?
            .to_dtype(dtype)?;
        let conv_count = decoder.count_causal_conv3d();
        Ok(Self {
            config,
            post_quant_conv,
            decoder,
            latents_mean: mean,
            latents_std: std,
            feat_cache: FeatCache::new(conv_count),
            use_tiling: false,
            tile_sample_min_height: 256,
            tile_sample_min_width: 256,
            tile_sample_stride_height: 192,
            tile_sample_stride_width: 192,
        })
    }

    pub fn from_var_builder(
        config: WanVaeConfig,
        vb: VarBuilder,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let post_quant_conv = WanCausalConv3d::new(
            config.z_dim,
            config.z_dim,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            vb.pp("post_quant_conv"),
        )?;
        let decoder = WanDecoder3d::new(&config, vb.pp("decoder"))?;
        Self::new(config, post_quant_conv, decoder, device, dtype)
    }

    pub fn load(
        vae_dir: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, LoaderError> {
        let dir = vae_dir.as_ref();
        let config_path = dir.join("config.json");
        let config: WanVaeConfig =
            crate::models::ltx_video::loader::load_model_config(&config_path)?;
        let shards = discover_safetensors(dir)?;
        let loader = WeightLoader::new(device.clone(), dtype);
        let vb = if shards.len() == 1 {
            loader
                .load_single(&shards[0])
                .map_err(LoaderError::Candle)?
        } else {
            loader
                .load_sharded(shards.as_slice())
                .map_err(LoaderError::Candle)?
        };
        Self::from_var_builder(config, vb, device.clone(), dtype).map_err(LoaderError::Candle)
    }

    pub fn load_with_layout(
        layout: &WanLayout,
        config: &WanVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, LoaderError> {
        let vb = load_vae_var_builder(layout, device, dtype)?;
        Self::from_var_builder(config.clone(), vb, device.clone(), dtype)
            .map_err(LoaderError::Candle)
    }

    pub fn config(&self) -> &WanVaeConfig {
        &self.config
    }

    /// Diffusers `enable_tiling` defaults (256 min / 192 stride).
    pub fn enable_tiling(&mut self) {
        self.use_tiling = true;
    }

    /// Larger tiles for 12 GB GPUs (fewer decoder passes; falls back to 256 on OOM).
    pub fn enable_large_tiling(&mut self) {
        self.use_tiling = true;
        self.tile_sample_min_height = 512;
        self.tile_sample_min_width = 512;
        self.tile_sample_stride_height = 448;
        self.tile_sample_stride_width = 448;
    }

    pub fn denormalize_latents(&self, z: &Tensor) -> Result<Tensor> {
        let z = z.broadcast_mul(&self.latents_std)?;
        z.broadcast_add(&self.latents_mean)
    }

    pub fn clear_cache(&mut self) {
        self.feat_cache.clear();
    }

    /// Decode normalized latents `[B, C, T, H, W]` to pixels in `[-1, 1]`.
    pub fn decode(&mut self, z: &Tensor) -> Result<Tensor> {
        let dtype = self.latents_mean.dtype();
        let z = z.to_dtype(dtype)?;
        let z = self.denormalize_latents(&z)?;
        let _ = z.dims5()?;

        if self.use_tiling {
            return self.tiled_decode(&z);
        }
        match self.decode_full(&z) {
            Ok(out) => Ok(out),
            Err(e) if is_cuda_oom(&e) => self.tiled_decode_adaptive(&z),
            Err(e) => Err(e),
        }
    }

    /// Try 512px tiles first (fits ~4 GB peak on 3060), then Diffusers 256px defaults.
    fn tiled_decode_adaptive(&mut self, z: &Tensor) -> Result<Tensor> {
        let saved = (
            self.tile_sample_min_height,
            self.tile_sample_min_width,
            self.tile_sample_stride_height,
            self.tile_sample_stride_width,
        );
        self.enable_large_tiling();
        match self.tiled_decode(z) {
            Ok(out) => Ok(out),
            Err(e) if is_cuda_oom(&e) => {
                self.tile_sample_min_height = saved.0.min(256);
                self.tile_sample_min_width = saved.1.min(256);
                self.tile_sample_stride_height = saved.2.min(192);
                self.tile_sample_stride_width = saved.3.min(192);
                self.tiled_decode(z)
            }
            Err(e) => Err(e),
        }
    }

    fn decode_full(&mut self, z: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(z, None)?;
        let t = z.dim(2)?;
        self.clear_cache();

        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            self.feat_cache.reset_idx();
            let frame = x.narrow(2, i, 1)?;
            let out = self.decoder.forward(&frame, &mut self.feat_cache)?;
            outs.push(out);
        }

        self.clear_cache();
        let out = Tensor::cat(&outs, 2)?;
        out.clamp(-1.0, 1.0)
    }

    fn tiled_decode(&mut self, z: &Tensor) -> Result<Tensor> {
        let (_, _, num_frames, height, width) = z.dims5()?;
        let sample_height = height * SPATIAL_COMPRESSION;
        let sample_width = width * SPATIAL_COMPRESSION;

        let tile_latent_min_h = self.tile_sample_min_height / SPATIAL_COMPRESSION;
        let tile_latent_min_w = self.tile_sample_min_width / SPATIAL_COMPRESSION;
        let tile_latent_stride_h = self.tile_sample_stride_height / SPATIAL_COMPRESSION;
        let tile_latent_stride_w = self.tile_sample_stride_width / SPATIAL_COMPRESSION;
        let blend_height = self.tile_sample_min_height - self.tile_sample_stride_height;
        let blend_width = self.tile_sample_min_width - self.tile_sample_stride_width;

        // ponytail: one post_quant for full latent volume; narrow tiles from x (not 315× post_quant).
        let x = self.post_quant_conv.forward(z, None)?;

        let mut rows = Vec::new();
        let mut i = 0usize;
        while i < height {
            let h_len = tile_latent_min_h.min(height - i);
            let mut row = Vec::new();
            let mut j = 0usize;
            while j < width {
                let w_len = tile_latent_min_w.min(width - j);
                self.clear_cache();
                let mut time = Vec::with_capacity(num_frames);
                for k in 0..num_frames {
                    self.feat_cache.reset_idx();
                    let tile = x
                        .narrow(2, k, 1)?
                        .narrow(3, i, h_len)?
                        .narrow(4, j, w_len)?;
                    let decoded = self.decoder.forward(&tile, &mut self.feat_cache)?;
                    time.push(decoded);
                }
                row.push(Tensor::cat(&time, 2)?);
                j += tile_latent_stride_w;
            }
            rows.push(row);
            i += tile_latent_stride_h;
        }
        self.clear_cache();

        let device = z.device();
        let blend_h_w = blend_weights(blend_height, device, 3)?;
        let blend_h_w2 = blend_weights(blend_width, device, 4)?;

        let mut result_rows = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row = Vec::with_capacity(row.len());
            for (rj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();
                if ri > 0 {
                    tile = blend_v(&rows[ri - 1][rj], &tile, blend_height, &blend_h_w)?;
                }
                if rj > 0 {
                    tile = blend_h(&row[rj - 1], &tile, blend_width, &blend_h_w2)?;
                }
                let stride_h = self.tile_sample_stride_height.min(tile.dim(3)?);
                let stride_w = self.tile_sample_stride_width.min(tile.dim(4)?);
                result_row.push(tile.narrow(3, 0, stride_h)?.narrow(4, 0, stride_w)?);
            }
            result_rows.push(Tensor::cat(&result_row, 4)?);
        }

        let out = Tensor::cat(&result_rows, 3)?
            .narrow(3, 0, sample_height)?
            .narrow(4, 0, sample_width)?;
        out.clamp(-1.0, 1.0)
    }
}

fn is_cuda_oom(err: &candle_core::Error) -> bool {
    format!("{err}").contains("OUT_OF_MEMORY")
}

fn blend_weights(blend: usize, device: &Device, dim: usize) -> Result<(Tensor, Tensor)> {
    if blend == 0 {
        let one = Tensor::ones((), DType::F32, device)?;
        return Ok((one.clone(), one));
    }
    let w = Tensor::arange(0u32, blend as u32, device)?
        .to_dtype(DType::F32)?
        .affine(1.0 / blend as f64, 0.0)?;
    let mut shape = [1usize; 5];
    shape[dim] = blend;
    let w = w.reshape(&shape)?;
    let one_minus = w.neg()?.affine(1.0, 1.0)?;
    Ok((w, one_minus))
}

fn blend_h(
    a: &Tensor,
    b: &Tensor,
    blend_extent: usize,
    weights: &(Tensor, Tensor),
) -> Result<Tensor> {
    let blend = blend_extent.min(a.dim(4)?).min(b.dim(4)?);
    if blend == 0 {
        return Ok(b.clone());
    }
    let (w, one_minus) = weights;
    let w = w.narrow(4, 0, blend)?.to_dtype(b.dtype())?;
    let one_minus = one_minus.narrow(4, 0, blend)?.to_dtype(b.dtype())?;

    let b_head = b.narrow(4, 0, blend)?;
    let b_tail = b.narrow(4, blend, b.dim(4)? - blend)?;
    let aw = a.dim(4)?;
    let a_tail = a.narrow(4, aw - blend, blend)?;

    let mixed = a_tail
        .broadcast_mul(&one_minus)?
        .add(&b_head.broadcast_mul(&w)?)?;
    Tensor::cat(&[&mixed, &b_tail], 4)
}

fn blend_v(
    a: &Tensor,
    b: &Tensor,
    blend_extent: usize,
    weights: &(Tensor, Tensor),
) -> Result<Tensor> {
    let blend = blend_extent.min(a.dim(3)?).min(b.dim(3)?);
    if blend == 0 {
        return Ok(b.clone());
    }
    let (w, one_minus) = weights;
    let w = w.narrow(3, 0, blend)?.to_dtype(b.dtype())?;
    let one_minus = one_minus.narrow(3, 0, blend)?.to_dtype(b.dtype())?;

    let b_head = b.narrow(3, 0, blend)?;
    let b_tail = b.narrow(3, blend, b.dim(3)? - blend)?;
    let ah = a.dim(3)?;
    let a_tail = a.narrow(3, ah - blend, blend)?;

    let mixed = a_tail
        .broadcast_mul(&one_minus)?
        .add(&b_head.broadcast_mul(&w)?)?;
    Tensor::cat(&[&mixed, &b_tail], 3)
}

/// Reference latent denormalization for tests: `z / std + mean`.
#[allow(dead_code)]
pub fn denormalize_latents_vec(z: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    z.iter()
        .enumerate()
        .map(|(i, &v)| v * std[i % std.len()] + mean[i % mean.len()])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_h_identity_when_zero_extent() {
        let device = Device::Cpu;
        let a = Tensor::zeros((1, 3, 1, 4, 4), DType::F32, &device).unwrap();
        let b = Tensor::ones((1, 3, 1, 4, 4), DType::F32, &device).unwrap();
        let w = blend_weights(0, &device, 4).unwrap();
        let out = blend_h(&a, &b, 0, &w).unwrap();
        assert_eq!(out.to_vec3::<f32>().unwrap()[0][0][0], 1.0);
    }
}
