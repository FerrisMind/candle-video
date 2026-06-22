//! Save generated video tensors to PNG frames or GIF.

use std::fs::File;
use std::path::Path;

use candle_core::{DType, IndexOp, Result, Tensor};
use gif::{Encoder, Repeat};
use rayon::prelude::*;

use crate::engine::video::VideoOutput;

/// Pixel value range of the input tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelRange {
    /// Values in `[0, 255]` (LTX pipeline output).
    ZeroTo255,
    /// Values in `[-1, 1]` (Wan VAE decode output).
    NegOneToOne,
}

/// Convert a `[B, C, T, H, W]` video tensor into RGB8 frame buffers.
pub fn tensor_to_rgb_frames(
    frames: &Tensor,
    pixel_range: PixelRange,
) -> Result<(Vec<Vec<u8>>, usize, usize)> {
    let (b, _c, f, h, w) = frames.dims5()?;
    let mut frame_data = Vec::with_capacity(b * f);

    for i in 0..b {
        for j in 0..f {
            let frame = frames.i((i, .., j, .., ..))?;
            let frame = frame.permute((1, 2, 0))?;
            let frame = match pixel_range {
                PixelRange::ZeroTo255 => frame.clamp(0.0, 255.0)?,
                PixelRange::NegOneToOne => frame.affine(127.5, 127.5)?.clamp(0.0, 255.0)?,
            };
            let frame = frame.to_dtype(DType::U8)?;
            frame_data.push(frame.flatten_all()?.to_vec1()?);
        }
    }

    Ok((frame_data, w, h))
}

/// Write PNG frames to `output_dir/frame_XXXX.png`.
pub fn save_png_frames(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    for (idx, data) in frame_data.iter().enumerate() {
        let path = output_dir.join(format!("frame_{idx:04}.png"));
        image::save_buffer(
            &path,
            data,
            width as u32,
            height as u32,
            image::ColorType::Rgb8,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(())
}

/// Write an animated GIF from RGB8 frame buffers.
pub fn save_gif(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    output_path: impl AsRef<Path>,
    frame_delay: u16,
) -> Result<()> {
    let output_path = output_path.as_ref();
    let mut file = File::create(output_path).map_err(candle_core::Error::wrap)?;
    let mut encoder = Encoder::new(&mut file, width as u16, height as u16, &[])
        .map_err(candle_core::Error::wrap)?;
    encoder
        .set_repeat(Repeat::Infinite)
        .map_err(candle_core::Error::wrap)?;

    let gif_frames: Vec<_> = frame_data
        .par_iter()
        .map(|data| {
            let mut frame = gif::Frame::from_rgb_speed(width as u16, height as u16, data, 30);
            frame.delay = frame_delay;
            frame
        })
        .collect();

    for frame in gif_frames {
        encoder
            .write_frame(&frame)
            .map_err(candle_core::Error::wrap)?;
    }
    Ok(())
}

/// Export [`VideoOutput`] to PNG frames and/or GIF.
pub fn export_video_output(
    output: &VideoOutput,
    output_dir: impl AsRef<Path>,
    pixel_range: PixelRange,
    save_png: bool,
    write_gif: bool,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).map_err(candle_core::Error::wrap)?;
    }

    let (frame_data, w, h) = tensor_to_rgb_frames(&output.frames, pixel_range)?;

    if save_png {
        save_png_frames(&frame_data, w, h, output_dir)?;
    }

    if write_gif {
        let delay = (100.0 / output.frame_rate.max(1) as f32) as u16;
        save_gif(
            &frame_data,
            w,
            h,
            output_dir.join("video.gif"),
            delay.max(1),
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn neg_one_to_one_maps_to_rgb() {
        let device = Device::Cpu;
        let frames = Tensor::new(&[-1.0f32, 0.0, 1.0], &device)
            .expect("t")
            .reshape((1, 1, 1, 1, 3))
            .expect("shape");
        let (data, w, h) = tensor_to_rgb_frames(&frames, PixelRange::NegOneToOne).expect("convert");
        assert_eq!((w, h), (3, 1));
        assert_eq!(data[0], vec![0, 127, 255]);
    }
}
