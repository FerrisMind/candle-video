use std::path::Path;

use candle_core::{DType, Device, Error, Result};

use crate::interfaces::conditioning::Conditioning;
use crate::interfaces::processor::{VideoInput, VideoOutput};
use crate::interfaces::video_types::VideoLatents;

pub type PipelineInput = VideoInput;
pub type PipelineOutput = VideoOutput;

pub trait DiffusionPipeline {
    fn register_modules(&mut self, _modules: &[&str]) {}

    fn from_pretrained<P: AsRef<Path>>(_path: P) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Msg("from_pretrained is not implemented".to_string()))
    }

    fn save_pretrained<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        Err(Error::Msg("save_pretrained is not implemented".to_string()))
    }

    fn to(&mut self, _device: &Device, _dtype: Option<DType>) -> Result<()> {
        Ok(())
    }
}

pub trait PipelineInference {
    fn encode_prompt(
        &mut self,
        _prompt: &str,
        _negative: Option<&str>,
        _device: &Device,
    ) -> Result<Option<Conditioning>> {
        Ok(None)
    }

    fn check_inputs(&self, _height: usize, _width: usize, _num_frames: usize) -> Result<()> {
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_latents(
        &self,
        batch_size: usize,
        num_channels_latents: usize,
        height: usize,
        width: usize,
        num_frames: usize,
        dtype: DType,
        device: &Device,
        latents: Option<VideoLatents>,
    ) -> Result<VideoLatents>;

    fn guidance_scale(&self) -> f64;

    fn do_classifier_free_guidance(&self) -> bool;

    fn num_timesteps(&self) -> usize;
}

pub fn apply_pipeline_io<P: DiffusionPipeline>(pipeline: &mut P) {
    if let Ok(path) = std::env::var("CANDLE_VIDEO_PIPELINE_LOAD_PATH") {
        let _ = P::from_pretrained(path);
    }
    if let Ok(path) = std::env::var("CANDLE_VIDEO_PIPELINE_SAVE_PATH") {
        let _ = pipeline.save_pretrained(path);
    }
    if std::env::var("CANDLE_VIDEO_PIPELINE_TO_CPU").is_ok() {
        let _ = pipeline.to(&Device::Cpu, None);
    }
}
