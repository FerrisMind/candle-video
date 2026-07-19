//! High-level video generation pipeline trait and request types.

use std::sync::Arc;

use candle_core::{Device, Result};

use super::memory::MemoryOptions;
use super::model::ModelCapabilities;
use super::progress::{NoopProgressObserver, ProgressObserver};
use super::video::{OutputOptions, VideoOutput, VideoTask};

/// User-facing generation request shared across backends.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub steps: usize,
    pub guidance_scale: f32,
    pub guidance_rescale: f32,
    pub seed: Option<u64>,
    pub frame_rate: usize,
    pub task: VideoTask,
    pub memory: MemoryOptions,
    pub output: OutputOptions,
    /// Pre-packed initial latents for reproducibility (backend-specific layout).
    pub initial_latents: Option<candle_core::Tensor>,
    /// Pre-computed prompt embeddings bypassing the text encoder.
    pub prompt_embeds: Option<candle_core::Tensor>,
    pub prompt_attention_mask: Option<candle_core::Tensor>,
    pub negative_prompt_embeds: Option<candle_core::Tensor>,
    pub negative_prompt_attention_mask: Option<candle_core::Tensor>,
}

impl GenerateRequest {
    pub fn text_to_video(
        prompt: impl Into<String>,
        width: usize,
        height: usize,
        frames: usize,
        steps: usize,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            width,
            height,
            frames,
            steps,
            guidance_scale: 5.0,
            guidance_rescale: 0.0,
            seed: None,
            frame_rate: 16,
            task: VideoTask::TextToVideo,
            memory: MemoryOptions::default(),
            output: OutputOptions::default(),
            initial_latents: None,
            prompt_embeds: None,
            prompt_attention_mask: None,
            negative_prompt_embeds: None,
            negative_prompt_attention_mask: None,
        }
    }
}

/// Materialize the seed used for a generation request.
///
/// The official Wan implementation treats its negative sentinel as “choose a
/// random integer in `0..=sys.maxsize`”. Keep the same signed-64-bit range for
/// omitted CLI seeds while preserving explicit values for reproducibility.
pub fn resolve_seed(seed: Option<u64>) -> u64 {
    match seed {
        Some(seed) => seed,
        None => rand::random_range(0..=i64::MAX as u64),
    }
}

#[cfg(test)]
fn resolve_seed_with_fallback(seed: Option<u64>, fallback: u64) -> u64 {
    seed.unwrap_or(fallback)
}

/// Unified pipeline interface for all video generation backends.
pub trait VideoPipeline {
    fn capabilities(&self) -> ModelCapabilities;
    fn validate(&self, req: &GenerateRequest) -> Result<()>;
    fn generate(&mut self, req: GenerateRequest, device: &Device) -> Result<VideoOutput>;

    /// Generate while publishing backend-neutral progress events.
    ///
    /// Existing third-party backends retain source compatibility through the
    /// default implementation; Wan and LTX override it to instrument their
    /// native denoising loops.
    fn generate_with_observer(
        &mut self,
        req: GenerateRequest,
        device: &Device,
        observer: Arc<dyn ProgressObserver>,
    ) -> Result<VideoOutput> {
        let _ = observer;
        self.generate(req, device)
    }

    /// Convenience wrapper for callers that do not need progress events.
    fn generate_without_progress(
        &mut self,
        req: GenerateRequest,
        device: &Device,
    ) -> Result<VideoOutput> {
        self.generate_with_observer(req, device, Arc::new(NoopProgressObserver))
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_seed_with_fallback;

    #[test]
    fn explicit_seed_is_preserved() {
        assert_eq!(resolve_seed_with_fallback(Some(42), 99), 42);
    }

    #[test]
    fn missing_seed_uses_the_materialized_random_seed() {
        assert_eq!(resolve_seed_with_fallback(None, 99), 99);
    }
}
