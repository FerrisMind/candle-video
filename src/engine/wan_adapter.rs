//! Wan backend adapter implementing the unified [`VideoPipeline`] trait.

use candle_core::{Device, Result};

use crate::models::wan::{WanGenerateRequest, WanPipeline};

use super::model::ModelCapabilities;
use super::pipeline::{GenerateRequest, VideoPipeline};
use super::video::{VideoOutput, VideoTask};

/// Wraps [`WanPipeline`] for the unified engine interface.
pub struct WanPipelineAdapter {
    pipeline: WanPipeline,
}

impl WanPipelineAdapter {
    pub fn new(pipeline: WanPipeline) -> Self {
        Self { pipeline }
    }

    pub fn pipeline(&self) -> &WanPipeline {
        &self.pipeline
    }

    pub fn pipeline_mut(&mut self) -> &mut WanPipeline {
        &mut self.pipeline
    }
}

impl VideoPipeline for WanPipelineAdapter {
    fn capabilities(&self) -> ModelCapabilities {
        self.pipeline.capabilities()
    }

    fn validate(&self, req: &GenerateRequest) -> Result<()> {
        if req.task != VideoTask::TextToVideo {
            candle_core::bail!("Wan adapter only supports TextToVideo, got {:?}", req.task);
        }

        let wan_req = WanGenerateRequest {
            prompt: Some(req.prompt.clone()),
            negative_prompt: req.negative_prompt.clone(),
            height: req.height,
            width: req.width,
            num_frames: req.frames,
            num_inference_steps: req.steps,
            guidance_scale: req.guidance_scale,
            seed: req.seed,
            max_sequence_length: crate::models::wan::WAN_DEFAULT_MAX_SEQ_LEN,
            initial_latents: req.initial_latents.clone(),
            prompt_embeds: req.prompt_embeds.clone(),
            negative_prompt_embeds: req.negative_prompt_embeds.clone(),
        };

        self.pipeline.check_inputs(&wan_req)
    }

    fn generate(&mut self, req: GenerateRequest, _device: &Device) -> Result<VideoOutput> {
        self.validate(&req)?;

        let wan_req = WanGenerateRequest {
            prompt: if req.prompt_embeds.is_some() {
                None
            } else {
                Some(req.prompt)
            },
            negative_prompt: req.negative_prompt,
            height: req.height,
            width: req.width,
            num_frames: req.frames,
            num_inference_steps: req.steps,
            guidance_scale: req.guidance_scale,
            seed: req.seed,
            max_sequence_length: crate::models::wan::WAN_DEFAULT_MAX_SEQ_LEN,
            initial_latents: req.initial_latents,
            prompt_embeds: req.prompt_embeds,
            negative_prompt_embeds: req.negative_prompt_embeds,
        };

        let out = self.pipeline.generate(wan_req)?;
        Ok(VideoOutput {
            frames: out.frames,
            frame_rate: req.frame_rate,
            width: req.width,
            height: req.height,
        })
    }
}
