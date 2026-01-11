use candle_core::{Result, Tensor};

use crate::interfaces::video_types::VideoLatents;

#[derive(Debug)]
pub enum VideoInput {
    Latents(VideoLatents),
    Image(image::DynamicImage),
    Video(Vec<image::DynamicImage>),
}

#[derive(Debug)]
pub enum VideoOutput {
    Tensor(Tensor),
    Frames(Vec<image::RgbImage>),
}

pub trait VideoProcessor {
    fn preprocess_video(&self, input: &VideoInput) -> Result<Tensor>;
    fn postprocess_video(&self, video: Tensor) -> Result<VideoOutput>;

    fn resize_and_crop_tensor(
        &self,
        tensor: &Tensor,
        target_height: usize,
        target_width: usize,
    ) -> Result<Tensor> {
        let (_, _, height, width) = tensor.dims4()?;
        if height == target_height && width == target_width {
            return Ok(tensor.clone());
        }

        let scale_h = target_height as f64 / height as f64;
        let scale_w = target_width as f64 / width as f64;
        let scale = scale_h.max(scale_w);
        let resized_h = (height as f64 * scale).round() as usize;
        let resized_w = (width as f64 * scale).round() as usize;

        let resized = tensor.interpolate2d(resized_h, resized_w)?;
        let top = (resized_h.saturating_sub(target_height)) / 2;
        let left = (resized_w.saturating_sub(target_width)) / 2;

        resized
            .narrow(2, top, target_height)?
            .narrow(3, left, target_width)
    }
}
