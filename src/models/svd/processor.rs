use candle_core::{DType, Device, Result, Tensor};

use crate::interfaces::pipeline::PipelineInput;
use crate::interfaces::processor::{VideoInput, VideoOutput, VideoProcessor};

#[derive(Debug, Clone)]
pub struct SvdVideoProcessor {
    device: Device,
    dtype: DType,
}

impl SvdVideoProcessor {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype }
    }
}

impl VideoProcessor for SvdVideoProcessor {
    fn preprocess_video(&self, input: &VideoInput) -> Result<Tensor> {
        let input: &PipelineInput = input;
        match input {
            VideoInput::Latents(latents) => Ok(latents.tensor.clone()),
            VideoInput::Image(image) => {
                let rgb = image.to_rgb8();
                let (width, height) = rgb.dimensions();
                let data: Vec<f32> = rgb
                    .pixels()
                    .flat_map(|p| {
                        let [r, g, b] = p.0;
                        [
                            (r as f32 / 255.0) * 2.0 - 1.0,
                            (g as f32 / 255.0) * 2.0 - 1.0,
                            (b as f32 / 255.0) * 2.0 - 1.0,
                        ]
                    })
                    .collect();

                Tensor::from_vec(data, (height as usize, width as usize, 3), &self.device)?
                    .permute((2, 0, 1))?
                    .unsqueeze(0)?
                    .to_dtype(self.dtype)
            }
            VideoInput::Video(_) => {
                candle_core::bail!("SVD video processor does not support video inputs")
            }
        }
    }

    fn postprocess_video(&self, video: Tensor) -> Result<VideoOutput> {
        Ok(VideoOutput::Tensor(video))
    }
}
