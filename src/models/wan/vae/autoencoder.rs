//! `AutoencoderKLWan` decode-only wrapper.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::ltx_video::loader::{LoaderError, WeightLoader};
use crate::models::wan::configs::WanVaeConfig;
use crate::models::wan::loader::discover_safetensors;

use super::causal_conv3d::{FeatCache, WanCausalConv3d};
use super::decoder::WanDecoder3d;

/// Wan 2.1 VAE (decode path only).
pub struct AutoencoderKLWan {
    config: WanVaeConfig,
    post_quant_conv: WanCausalConv3d,
    decoder: WanDecoder3d,
    latents_mean: Tensor,
    latents_std: Tensor,
    feat_cache: FeatCache,
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

    pub fn load(vae_dir: impl AsRef<Path>, device: &Device, dtype: DType) -> std::result::Result<Self, LoaderError> {
        let dir = vae_dir.as_ref();
        let config_path = dir.join("config.json");
        let config: WanVaeConfig =
            crate::models::ltx_video::loader::load_model_config(&config_path)?;
        let shards = discover_safetensors(dir)?;
        let loader = WeightLoader::new(device.clone(), dtype);
        let vb = if shards.len() == 1 {
            loader.load_single(&shards[0]).map_err(LoaderError::Candle)?
        } else {
            loader
                .load_sharded(shards.as_slice())
                .map_err(LoaderError::Candle)?
        };
        Self::from_var_builder(config, vb, device.clone(), dtype).map_err(LoaderError::Candle)
    }

    pub fn config(&self) -> &WanVaeConfig {
        &self.config
    }

    pub fn denormalize_latents(&self, z: &Tensor) -> Result<Tensor> {
        // Diffusers: latents / (1/std) + mean == latents * std + mean
        let z = z.broadcast_mul(&self.latents_std)?;
        z.broadcast_add(&self.latents_mean)
    }

    pub fn clear_cache(&mut self) {
        self.feat_cache.clear();
    }

    /// Decode normalized latents `[B, C, T, H, W]` to pixels in `[-1, 1]`.
    pub fn decode(&mut self, z: &Tensor) -> Result<Tensor> {
        let z = self.denormalize_latents(z)?;
        let x = self.post_quant_conv.forward(&z, None)?;
        let t = z.dim(2)?;
        self.clear_cache();

        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            self.feat_cache.reset_idx();
            let frame = x.narrow(2, i, 1)?;
            let first_chunk = i == 0;
            let _first_chunk = first_chunk;
            let out = self.decoder.forward(&frame, &mut self.feat_cache)?;
            outs.push(out);
        }

        self.clear_cache();
        let out = Tensor::cat(&outs, 2)?;
        out.clamp(-1.0, 1.0)
    }
}

/// Reference latent denormalization for tests: `z / std + mean`.
#[allow(dead_code)]
pub fn denormalize_latents_vec(
    z: &[f32],
    mean: &[f32],
    std: &[f32],
) -> Vec<f32> {
    z.iter()
        .enumerate()
        .map(|(i, &v)| v * std[i % std.len()] + mean[i % mean.len()])
        .collect()
}
