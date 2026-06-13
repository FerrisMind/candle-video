//! Video output types and resolution constraints.

use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// Supported generation task kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoTask {
    TextToVideo,
    ImageToVideo,
    TextImageToVideo,
}

/// A width/height pair in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

/// Frame-count constraints for a model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRule {
    pub min_frames: usize,
    pub max_frames: usize,
    /// Frames must satisfy `(n - 1) % step == 0` when set.
    pub step: Option<usize>,
}

/// Options for writing generated video to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputOptions {
    pub frame_rate: usize,
    pub save_frames: bool,
    pub save_gif: bool,
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self {
            frame_rate: 16,
            save_frames: false,
            save_gif: false,
        }
    }
}

/// Tensor video output from a generation pipeline.
#[derive(Debug, Clone)]
pub struct VideoOutput {
    pub frames: Tensor,
    pub frame_rate: usize,
    pub width: usize,
    pub height: usize,
}

impl VideoOutput {
    pub fn num_frames(&self) -> candle_core::Result<usize> {
        // [B, C, F, H, W] or [B, F, H, W, C] depending on backend postprocess
        let dims = self.frames.dims();
        match dims.len() {
            5 => Ok(dims[2]),
            4 => Ok(dims[1]),
            _ => candle_core::bail!(
                "VideoOutput expected 4D or 5D tensor, got rank {}",
                dims.len()
            ),
        }
    }
}
