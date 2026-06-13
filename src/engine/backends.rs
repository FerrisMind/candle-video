//! Low-level backend traits for individual model components.

use candle_core::{DType, Device, Result, Tensor};

use super::prompt::PromptEmbeds;

/// Text encoder component (T5, UMT5, CLIP, etc.).
pub trait TextEncoderBackend: Send {
    fn dtype(&self) -> DType;
    fn forward(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;
}

/// Diffusion transformer operating on video latents.
pub trait VideoTransformerBackend: Send {
    fn in_channels(&self) -> usize;
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}

/// Video VAE decode (and optionally encode) backend.
pub trait VideoVaeBackend: Send {
    fn dtype(&self) -> DType;
    fn latent_channels(&self) -> usize;
    fn spatial_compression_ratio(&self) -> usize;
    fn temporal_compression_ratio(&self) -> usize;
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}

/// Noise scheduler stepping backend.
pub trait SchedulerBackend: Send {
    fn set_timesteps(&mut self, steps: usize, device: &Device) -> Result<Vec<i64>>;
    fn step(&mut self, model_output: &Tensor, timestep: i64, latents: &Tensor) -> Result<Tensor>;
}

/// Weight discovery and loading for a Diffusers-style folder layout.
pub trait WeightLoaderBackend: Send {
    fn model_root(&self) -> &std::path::Path;
    fn discover_components(&self) -> Result<()>;
}

/// Convenience bundle for backends that encode prompts end-to-end.
pub trait PromptEncodingBackend: Send {
    fn encode_prompt(&mut self, prompt: &str, max_len: usize) -> Result<PromptEmbeds>;
}
