pub mod decoder;
pub mod encoder;

pub use decoder::TemporalDecoder;
pub use encoder::Encoder;

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::config::SvdVaeConfig;
use crate::interfaces::autoencoder::{AutoencoderError, VideoAutoencoder};
use crate::interfaces::autoencoder_mixin::AutoencoderMixin;
use crate::interfaces::distributions::DiagonalGaussian;
use crate::interfaces::video_types::{VideoLatents, VideoLayout};

pub type GaussianDistribution = DiagonalGaussian;

#[derive(Debug)]
pub struct AutoencoderKLTemporalDecoder {
    encoder: Encoder,

    quant_conv: candle_nn::Conv2d,

    temporal_decoder: TemporalDecoder,
    config: SvdVaeConfig,
}

impl AutoencoderKLTemporalDecoder {
    pub fn new(vb: VarBuilder, config: &SvdVaeConfig) -> Result<Self> {
        let encoder = Encoder::new(vb.pp("encoder"), config)?;

        let quant_conv = candle_nn::conv2d(
            2 * config.latent_channels,
            2 * config.latent_channels,
            1,
            Default::default(),
            vb.pp("quant_conv"),
        )?;

        let temporal_decoder = TemporalDecoder::new(vb.pp("decoder"), config)?;

        Ok(Self {
            encoder,
            quant_conv,
            temporal_decoder,
            config: config.clone(),
        })
    }

    pub fn encode(&self, x: &Tensor) -> Result<GaussianDistribution> {
        let x = if self.config.force_upcast && x.dtype() == DType::F16 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };

        let h = self.encoder.forward(&x)?;
        let h = self.quant_conv.forward(&h)?;
        GaussianDistribution::new(&h)
    }

    pub fn encode_to_latent(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();

        let x = if self.config.force_upcast && original_dtype == DType::F16 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };

        let posterior = self.encode(&x)?;
        let z = posterior.sample()?;
        let z = z.affine(self.config.scaling_factor, 0.0)?;

        if self.config.force_upcast && original_dtype == DType::F16 {
            z.to_dtype(DType::F16)
        } else {
            Ok(z)
        }
    }

    pub fn decode(
        &self,
        z: &Tensor,
        num_frames: usize,
        chunk_size: Option<usize>,
    ) -> Result<Tensor> {
        let original_dtype = z.dtype();
        let batch_frames = z.dim(0)?;
        let _batch_size = batch_frames / num_frames;
        let chunk_size = chunk_size.unwrap_or(num_frames);

        let z = if self.config.force_upcast && original_dtype == DType::F16 {
            z.to_dtype(DType::F32)?
        } else {
            z.clone()
        };

        let z = z.affine(1.0 / self.config.scaling_factor, 0.0)?;

        let mut decoded_chunks = Vec::new();
        for start in (0..batch_frames).step_by(chunk_size) {
            let end = std::cmp::min(start + chunk_size, batch_frames);
            let chunk_len = end - start;

            let z_chunk = z.narrow(0, start, chunk_len)?;

            let num_frames_in_chunk = chunk_len.min(num_frames);
            let batch_for_chunk = chunk_len.div_ceil(num_frames_in_chunk);
            let image_only_indicator = Tensor::zeros(
                (batch_for_chunk, num_frames_in_chunk),
                z.dtype(),
                z.device(),
            )?;

            let decoded = self.temporal_decoder.forward(
                &z_chunk,
                &image_only_indicator,
                num_frames_in_chunk,
            )?;
            decoded_chunks.push(decoded);
        }

        let decoded = if decoded_chunks.len() == 1 {
            decoded_chunks.remove(0)
        } else {
            Tensor::cat(&decoded_chunks, 0)?
        };

        if self.config.force_upcast && original_dtype == DType::F16 {
            decoded.to_dtype(DType::F16)
        } else {
            Ok(decoded)
        }
    }

    pub fn scaling_factor(&self) -> f64 {
        self.config.scaling_factor
    }
}

impl VideoAutoencoder for AutoencoderKLTemporalDecoder {
    fn decode(&self, latents: &VideoLatents) -> std::result::Result<Tensor, AutoencoderError> {
        let latents = latents.to_canonical()?;
        let flattened = latents.tensor.reshape((
            latents.batch * latents.frames,
            latents.channels,
            latents.height,
            latents.width,
        ))?;
        Ok(AutoencoderKLTemporalDecoder::decode(
            self,
            &flattened,
            latents.frames,
            None,
        )?)
    }

    fn encode(&self, video: &Tensor) -> std::result::Result<VideoLatents, AutoencoderError> {
        let latents = AutoencoderKLTemporalDecoder::encode_to_latent(self, video)?;
        let (b, c, h, w) = latents.dims4()?;
        Ok(VideoLatents {
            tensor: latents,
            layout: VideoLayout::BfCHW,
            batch: b,
            frames: 1,
            channels: c,
            height: h,
            width: w,
        })
    }
}

impl AutoencoderMixin for AutoencoderKLTemporalDecoder {
    fn enable_tiling(&mut self) {}

    fn disable_tiling(&mut self) {}

    fn enable_slicing(&mut self) {}

    fn disable_slicing(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_config() {
        let config = SvdVaeConfig::default();
        assert_eq!(config.latent_channels, 4);
        assert_eq!(config.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(config.scaling_factor, 0.18215);
        assert!(config.force_upcast);
    }
}
