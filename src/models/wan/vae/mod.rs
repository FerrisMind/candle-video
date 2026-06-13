//! Wan 2.1 `AutoencoderKLWan` decode path (Diffusers-compatible).

mod autoencoder;
mod blocks;
mod causal_conv3d;
mod decoder;
mod resample;
mod rms_norm;

pub use autoencoder::{AutoencoderKLWan, denormalize_latents_vec};
pub use causal_conv3d::{FeatCache, WanCausalConv3d, count_causal_conv3d};
pub use decoder::WanDecoder3d;
