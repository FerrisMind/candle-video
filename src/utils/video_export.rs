//! Save generated video tensors to PNG frames, GIF, or MP4.

use std::fs::File;
use std::path::Path;

use candle_core::{DType, IndexOp, Result, Tensor};
use gif::{Encoder, Repeat};
use muxide::api::{MuxerBuilder, VideoCodec};
use openh264::OpenH264API;
use openh264::encoder::{Encoder as H264Encoder, EncoderConfig, FrameType};
use openh264::formats::{RgbSliceU8, YUVBuffer};
use rayon::prelude::*;

use crate::engine::video::VideoOutput;
use crate::engine::{
    GenerationEvent, GenerationStage, NoopProgressObserver, ProgressObserver, ensure_not_cancelled,
};

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
            let frame = frame.round()?.to_dtype(DType::U8)?;
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
    save_png_frames_with_observer(frame_data, width, height, output_dir, None)
}

/// Write PNG frames while optionally publishing progress and cancellation.
pub fn save_png_frames_with_observer(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    output_dir: impl AsRef<Path>,
    observer: Option<&dyn ProgressObserver>,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir).map_err(candle_core::Error::wrap)?;
    for (idx, data) in frame_data.iter().enumerate() {
        if let Some(observer) = observer {
            ensure_not_cancelled(observer)?;
        }
        let path = output_dir.join(format!("frame_{idx:04}.png"));
        image::save_buffer(
            &path,
            data,
            width as u32,
            height as u32,
            image::ColorType::Rgb8,
        )
        .map_err(candle_core::Error::wrap)?;
        if let Some(observer) = observer {
            observer.on_event(&GenerationEvent::StageProgress {
                stage: GenerationStage::EncodeVideo,
                current: (idx + 1) as u64,
                total: Some(frame_data.len() as u64),
                message: Some(format!("{} / {} frames", idx + 1, frame_data.len())),
            });
        }
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
    save_gif_atomic(frame_data, width, height, output_path, frame_delay, false)
}

/// Encode a GIF through a sibling `.partial` file and publish it after the
/// animation has been finalized successfully.
pub fn save_gif_atomic(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    output_path: impl AsRef<Path>,
    frame_delay: u16,
    keep_partial: bool,
) -> Result<()> {
    save_gif_atomic_with_observer(
        frame_data,
        width,
        height,
        output_path,
        frame_delay,
        keep_partial,
        None,
    )
}

/// Atomic GIF export with optional per-frame progress and cancellation.
pub fn save_gif_atomic_with_observer(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    output_path: impl AsRef<Path>,
    frame_delay: u16,
    keep_partial: bool,
    observer: Option<&dyn ProgressObserver>,
) -> Result<()> {
    let output_path = output_path.as_ref();
    let partial_path = std::path::PathBuf::from(format!("{}.partial", output_path.display()));
    let result = write_gif_file(
        frame_data,
        width,
        height,
        &partial_path,
        frame_delay,
        observer,
    )
    .and_then(|_| {
        if output_path.exists() {
            std::fs::remove_file(output_path).map_err(candle_core::Error::wrap)?;
        }
        std::fs::rename(&partial_path, output_path).map_err(candle_core::Error::wrap)
    });

    if result.is_err() && !keep_partial {
        let _ = std::fs::remove_file(&partial_path);
    }
    result
}

fn write_gif_file(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    output_path: &Path,
    frame_delay: u16,
    observer: Option<&dyn ProgressObserver>,
) -> Result<()> {
    if frame_data.is_empty() {
        candle_core::bail!("GIF export requires at least one frame");
    }
    if width == 0 || height == 0 {
        candle_core::bail!("GIF dimensions must be positive, got {width}x{height}");
    }
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| candle_core::Error::Msg("GIF frame dimensions overflow".into()))?;
    for (index, frame) in frame_data.iter().enumerate() {
        if frame.len() != expected_len {
            candle_core::bail!(
                "GIF frame {} has {} RGB bytes, expected {} for {}x{}",
                index,
                frame.len(),
                expected_len,
                width,
                height
            );
        }
    }
    let width = u16::try_from(width)
        .map_err(|_| candle_core::Error::Msg(format!("GIF width {width} exceeds u16")))?;
    let height = u16::try_from(height)
        .map_err(|_| candle_core::Error::Msg(format!("GIF height {height} exceeds u16")))?;
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(candle_core::Error::wrap)?;
    }
    let mut file = File::create(output_path).map_err(candle_core::Error::wrap)?;
    let mut encoder =
        Encoder::new(&mut file, width, height, &[]).map_err(candle_core::Error::wrap)?;
    encoder
        .set_repeat(Repeat::Infinite)
        .map_err(candle_core::Error::wrap)?;

    let gif_frames: Vec<_> = frame_data
        .par_iter()
        .map(|data| {
            let mut frame = gif::Frame::from_rgb_speed(width, height, data, 30);
            frame.delay = frame_delay;
            frame
        })
        .collect();

    for (index, frame) in gif_frames.iter().enumerate() {
        if let Some(observer) = observer {
            ensure_not_cancelled(observer)?;
        }
        encoder
            .write_frame(frame)
            .map_err(candle_core::Error::wrap)?;
        if let Some(observer) = observer {
            observer.on_event(&GenerationEvent::StageProgress {
                stage: GenerationStage::EncodeVideo,
                current: (index + 1) as u64,
                total: Some(frame_data.len() as u64),
                message: Some(format!("{} / {} frames", index + 1, frame_data.len())),
            });
        }
    }
    Ok(())
}

/// Write RGB8 frames to a video-only H.264/MP4 file.
///
/// OpenH264 performs the RGB24 → I420 conversion in-process and `muxide`
/// writes the resulting Annex-B access units into a standards-compliant MP4
/// container.  No external encoder or process is invoked.  Width and height
/// must be even because H.264 4:2:0 uses two-by-two chroma blocks.
pub fn save_mp4(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: u32,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    save_mp4_atomic(frame_data, width, height, fps, output_path, false)
}

/// Encode an MP4 through a sibling `.partial` file and publish it after the
/// container has been finalized successfully.
pub fn save_mp4_atomic(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: u32,
    output_path: impl AsRef<Path>,
    keep_partial: bool,
) -> Result<()> {
    save_mp4_atomic_with_observer(
        frame_data,
        width,
        height,
        fps,
        output_path,
        keep_partial,
        None,
    )
}

/// Atomic MP4 export with optional per-frame progress events.
pub fn save_mp4_atomic_with_observer(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: u32,
    output_path: impl AsRef<Path>,
    keep_partial: bool,
    observer: Option<&dyn ProgressObserver>,
) -> Result<()> {
    let output_path = output_path.as_ref();
    let partial_path = std::path::PathBuf::from(format!("{}.partial", output_path.display()));
    let result =
        write_mp4_file(frame_data, width, height, fps, &partial_path, observer).and_then(|_| {
            if output_path.exists() {
                std::fs::remove_file(output_path).map_err(candle_core::Error::wrap)?;
            }
            std::fs::rename(&partial_path, output_path).map_err(candle_core::Error::wrap)
        });

    if result.is_err() && !keep_partial {
        let _ = std::fs::remove_file(&partial_path);
    }
    result
}

fn write_mp4_file(
    frame_data: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: u32,
    output_path: &Path,
    observer: Option<&dyn ProgressObserver>,
) -> Result<()> {
    if frame_data.is_empty() {
        candle_core::bail!("MP4 export requires at least one frame");
    }
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        candle_core::bail!(
            "MP4 dimensions must be positive even values for I420, got {}x{}",
            width,
            height
        );
    }
    if fps == 0 {
        candle_core::bail!("MP4 frame rate must be greater than zero");
    }
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| candle_core::Error::Msg("MP4 frame dimensions overflow".into()))?;
    for (index, frame) in frame_data.iter().enumerate() {
        if frame.len() != expected_len {
            candle_core::bail!(
                "MP4 frame {} has {} RGB bytes, expected {} for {}x{}",
                index,
                frame.len(),
                expected_len,
                width,
                height
            );
        }
    }

    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(candle_core::Error::wrap)?;
    }
    let file = File::create(output_path).map_err(candle_core::Error::wrap)?;
    let mut muxer = MuxerBuilder::new(file)
        .video(VideoCodec::H264, width as u32, height as u32, fps as f64)
        .build()
        .map_err(|error| candle_core::Error::Msg(format!("MP4 muxer setup failed: {error}")))?;
    // The OpenH264 default enables frame skipping for its low camera bitrate.
    // A generated frame must never disappear from a video, especially at
    // larger LTX resolutions, so disable rate-control frame dropping.
    let mut encoder = H264Encoder::with_api_config(
        OpenH264API::from_source(),
        EncoderConfig::new().skip_frames(false),
    )
    .map_err(|error| candle_core::Error::Msg(format!("OpenH264 setup failed: {error}")))?;

    for (index, rgb) in frame_data.iter().enumerate() {
        let rgb_source = RgbSliceU8::new(rgb, (width, height));
        let yuv = YUVBuffer::from_rgb8_source(rgb_source);
        let encoded = encoder.encode(&yuv).map_err(|error| {
            candle_core::Error::Msg(format!("OpenH264 frame {index} failed: {error}"))
        })?;
        let bytes = encoded.to_vec();
        if bytes.is_empty() {
            candle_core::bail!("OpenH264 produced an empty access unit for frame {}", index);
        }
        let keyframe = matches!(encoded.frame_type(), FrameType::IDR | FrameType::I);
        muxer
            .write_video(index as f64 / fps as f64, &bytes, keyframe)
            .map_err(|error| {
                candle_core::Error::Msg(format!("MP4 muxing frame {index} failed: {error}"))
            })?;
        if let Some(observer) = observer {
            observer.on_event(&GenerationEvent::StageProgress {
                stage: GenerationStage::EncodeVideo,
                current: (index + 1) as u64,
                total: Some(frame_data.len() as u64),
                message: Some(format!("{} / {} frames", index + 1, frame_data.len())),
            });
        }
    }

    muxer
        .finish()
        .map_err(|error| candle_core::Error::Msg(format!("MP4 finalization failed: {error}")))?;
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
    export_video_output_with_mp4(
        output,
        output_dir,
        pixel_range,
        save_png,
        write_gif,
        None,
        output.frame_rate,
    )
}

/// Export a [`VideoOutput`] and optionally encode an MP4 at `mp4_path`.
///
/// This extended entry point keeps the original PNG/GIF API source-compatible
/// while allowing callers such as the Wan CLI to choose an explicit output
/// filename and frame rate.
pub fn export_video_output_with_mp4(
    output: &VideoOutput,
    output_dir: impl AsRef<Path>,
    pixel_range: PixelRange,
    save_png: bool,
    write_gif: bool,
    mp4_path: Option<&Path>,
    fps: usize,
) -> Result<()> {
    export_video_output_with_mp4_options(
        output,
        output_dir,
        pixel_range,
        save_png,
        write_gif,
        mp4_path,
        fps,
        false,
    )
}

/// Export video frames and optionally publish an atomic MP4.
#[allow(clippy::too_many_arguments)]
pub fn export_video_output_with_mp4_options(
    output: &VideoOutput,
    output_dir: impl AsRef<Path>,
    pixel_range: PixelRange,
    save_png: bool,
    write_gif: bool,
    mp4_path: Option<&Path>,
    fps: usize,
    keep_partial: bool,
) -> Result<()> {
    let observer = NoopProgressObserver;
    export_video_output_with_mp4_options_and_observer(
        output,
        output_dir,
        pixel_range,
        save_png,
        write_gif,
        mp4_path,
        fps,
        keep_partial,
        &observer,
    )
}

/// Export video frames with an observer for machine-readable output progress.
#[allow(clippy::too_many_arguments)]
pub fn export_video_output_with_mp4_options_and_observer(
    output: &VideoOutput,
    output_dir: impl AsRef<Path>,
    pixel_range: PixelRange,
    save_png: bool,
    write_gif: bool,
    mp4_path: Option<&Path>,
    fps: usize,
    keep_partial: bool,
    observer: &dyn ProgressObserver,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).map_err(candle_core::Error::wrap)?;
    }

    let postprocess_started = std::time::Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::PostProcess,
        message: "Converting decoded tensor to RGB8 frames".to_string(),
    });
    let (frame_data, w, h) = tensor_to_rgb_frames(&output.frames, pixel_range)?;
    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::PostProcess,
        elapsed_secs: postprocess_started.elapsed().as_secs_f64(),
    });

    let encode_started = std::time::Instant::now();
    observer.on_event(&GenerationEvent::StageStarted {
        stage: GenerationStage::EncodeVideo,
        message: format!("Encoding {} frames", frame_data.len()),
    });

    if save_png {
        save_png_frames_with_observer(&frame_data, w, h, output_dir, Some(observer))?;
    }

    if write_gif {
        let delay = (100.0 / fps.max(1) as f32) as u16;
        let gif_path = output_dir.join("video.gif");
        save_gif_atomic_with_observer(
            &frame_data,
            w,
            h,
            &gif_path,
            delay.max(1),
            keep_partial,
            Some(observer),
        )?;
    }

    if let Some(mp4_path) = mp4_path {
        let fps = u32::try_from(fps)
            .map_err(|_| candle_core::Error::Msg(format!("FPS {fps} does not fit in u32")))?;
        save_mp4_atomic_with_observer(
            &frame_data,
            w,
            h,
            fps,
            mp4_path,
            keep_partial,
            Some(observer),
        )?;
    }

    observer.on_event(&GenerationEvent::StageFinished {
        stage: GenerationStage::EncodeVideo,
        elapsed_secs: encode_started.elapsed().as_secs_f64(),
    });

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
        assert_eq!(data[0], vec![0, 128, 255]);
    }

    #[test]
    fn save_mp4_writes_a_valid_h264_container() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("wan-smoke.mp4");
        let frames = vec![vec![0u8; 16 * 16 * 3], vec![255u8; 16 * 16 * 3]];

        save_mp4(&frames, 16, 16, 12, &path).expect("mp4 export");
        let bytes = std::fs::read(&path).expect("read mp4");
        assert!(bytes.len() > 32, "MP4 must contain a complete container");
        assert_eq!(&bytes[4..8], b"ftyp", "missing MP4 ftyp box");
        assert!(bytes.windows(4).any(|window| window == b"moov"));
        assert!(bytes.windows(4).any(|window| window == b"mdat"));
    }

    #[test]
    fn save_mp4_atomic_commits_and_removes_partial_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("atomic.mp4");
        let partial = std::path::PathBuf::from(format!("{}.partial", path.display()));
        let frames = vec![vec![0u8; 16 * 16 * 3]];

        save_mp4_atomic(&frames, 16, 16, 12, &path, false).expect("atomic mp4 export");

        assert!(path.exists(), "final MP4 must exist");
        assert!(
            !partial.exists(),
            "successful export must remove partial file"
        );
    }

    #[test]
    fn save_gif_atomic_commits_and_removes_partial_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("atomic.gif");
        let partial = std::path::PathBuf::from(format!("{}.partial", path.display()));
        let frames = vec![vec![0u8; 2 * 2 * 3]];

        save_gif_atomic(&frames, 2, 2, &path, 6, false).expect("atomic gif export");

        assert!(path.exists(), "final GIF must exist");
        assert!(
            !partial.exists(),
            "successful export must remove partial file"
        );
    }

    #[test]
    fn save_mp4_large_frames_do_not_get_skipped_by_rate_control() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("large.mp4");
        let width = 768;
        let height = 512;
        let frame_len = width * height * 3;
        let frames = vec![vec![96u8; frame_len], vec![160u8; frame_len]];

        save_mp4_atomic(&frames, width, height, 25, &path, false)
            .expect("large MP4 export must encode every frame");

        assert!(path.exists(), "large MP4 must be published");
    }
}
