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
