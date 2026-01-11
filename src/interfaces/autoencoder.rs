use candle_core::Tensor;

use crate::interfaces::autoencoder_mixin::AutoencoderMixin;
use crate::interfaces::video_types::VideoLatents;
#[derive(Debug, thiserror::Error)]
pub enum AutoencoderError {
    #[error("encode not supported")]
    EncodeNotSupported,
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

pub trait VideoAutoencoder: AutoencoderMixin {
    fn decode(&self, latents: &VideoLatents) -> Result<Tensor, AutoencoderError>;

    fn encode(&self, _video: &Tensor) -> Result<VideoLatents, AutoencoderError> {
        Err(AutoencoderError::EncodeNotSupported)
    }
}
