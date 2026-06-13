//! LTX-Video adapter implementing the unified [`VideoPipeline`] trait.

use candle_core::{Device, Result};

use crate::models::ltx_video::configs::LTXVInferenceConfig;
use crate::models::ltx_video::t2v_pipeline::{
    LtxPipeline, OutputType, PromptInput, Scheduler, TextEncoder, Tokenizer, VaeLtxVideo,
    VaeConfig, VideoProcessor, VideoTransformer3D,
};

use super::model::ModelCapabilities;
use super::pipeline::{GenerateRequest, VideoPipeline};
use super::video::{VideoOutput, VideoTask};

/// Wraps an existing [`LtxPipeline`] without changing LTX math or component traits.
pub struct LtxPipelineAdapter<'a> {
    pipeline: LtxPipeline<'a>,
    inference: LTXVInferenceConfig,
    version: String,
}

impl<'a> LtxPipelineAdapter<'a> {
    pub fn new(
        pipeline: LtxPipeline<'a>,
        inference: LTXVInferenceConfig,
        version: impl Into<String>,
    ) -> Self {
        Self {
            pipeline,
            inference,
            version: version.into(),
        }
    }

    pub fn pipeline_mut(&mut self) -> &mut LtxPipeline<'a> {
        &mut self.pipeline
    }

    pub fn pipeline(&self) -> &LtxPipeline<'a> {
        &self.pipeline
    }
}

impl VideoPipeline for LtxPipelineAdapter<'_> {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::ltx(&self.version)
    }

    fn validate(&self, req: &GenerateRequest) -> Result<()> {
        if req.task != VideoTask::TextToVideo {
            candle_core::bail!("LTX adapter only supports TextToVideo, got {:?}", req.task);
        }

        self.pipeline.check_inputs(
            Some(&PromptInput::Single(req.prompt.clone())),
            req.height,
            req.width,
            req.prompt_embeds.as_ref(),
            req.negative_prompt_embeds.as_ref(),
            req.prompt_attention_mask.as_ref(),
            req.negative_prompt_attention_mask.as_ref(),
        )
    }

    fn generate(&mut self, req: GenerateRequest, device: &Device) -> Result<VideoOutput> {
        self.validate(&req)?;

        let negative = req
            .negative_prompt
            .map(PromptInput::Single)
            .or_else(|| {
                if self.pipeline.do_classifier_free_guidance() {
                    Some(PromptInput::Single(String::new()))
                } else {
                    None
                }
            });

        let decode_timestep = self
            .inference
            .decode_timestep
            .clone()
            .unwrap_or_else(|| vec![0.05]);

        let output = self.pipeline.call(
            Some(PromptInput::Single(req.prompt)),
            negative,
            req.height,
            req.width,
            req.frames,
            req.frame_rate,
            req.steps,
            None,
            None,
            req.guidance_scale,
            req.guidance_rescale,
            self.inference.stg_scale,
            1,
            req.initial_latents,
            req.prompt_embeds,
            req.prompt_attention_mask,
            req.negative_prompt_embeds,
            req.negative_prompt_attention_mask,
            decode_timestep,
            self.inference.decode_noise_scale.clone(),
            OutputType::Tensor,
            self.pipeline.tokenizer_max_length,
            if self.inference.skip_block_list.is_empty() {
                None
            } else {
                Some(self.inference.skip_block_list.clone())
            },
            device,
        )?;

        Ok(VideoOutput {
            frames: output.frames,
            frame_rate: req.frame_rate,
            width: req.width,
            height: req.height,
        })
    }
}

/// Type alias preserving the concrete LTX component trait objects for builders.
pub type LtxPipelineParts<'a> = (
    Box<dyn Scheduler + 'a>,
    Box<dyn VaeLtxVideo + 'a>,
    Box<dyn TextEncoder + 'a>,
    Box<dyn Tokenizer + 'a>,
    Box<dyn VideoTransformer3D + 'a>,
    Box<dyn VideoProcessor + 'a>,
);

/// Build an [`LtxPipeline`] from owned component trait objects.
pub fn build_ltx_pipeline<'a>(parts: LtxPipelineParts<'a>) -> LtxPipeline<'a> {
    let (scheduler, vae, text_encoder, tokenizer, transformer, video_processor) = parts;
    LtxPipeline::new(
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        video_processor,
    )
}

/// Default VAE processor config for LTX pipelines.
pub fn default_ltx_vae_processor_config() -> VaeConfig {
    VaeConfig {
        scaling_factor: 1.0,
        timestep_conditioning: true,
    }
}
