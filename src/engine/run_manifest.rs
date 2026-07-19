//! Reproducibility metadata written alongside generated video files.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// Stable metadata schema for one generation run.
#[derive(Debug, Clone, Serialize)]
pub struct RunManifest {
    pub model: String,
    pub backend: String,
    pub device: String,
    pub dtype: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub scheduler: String,
    pub total_seconds: f64,
    pub peak_vram_bytes: Option<u64>,
    pub output: String,
}

/// Write a pretty JSON manifest through a temporary sibling and publish it.
pub fn write_run_manifest(
    path: impl AsRef<Path>,
    manifest: &RunManifest,
) -> candle_core::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(candle_core::Error::wrap)?;
    }
    let partial = PathBuf::from(format!("{}.partial", path.display()));
    let result = (|| {
        let bytes = serde_json::to_vec_pretty(manifest).map_err(candle_core::Error::wrap)?;
        std::fs::write(&partial, bytes).map_err(candle_core::Error::wrap)?;
        if path.exists() {
            std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        }
        std::fs::rename(&partial, path).map_err(candle_core::Error::wrap)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result
}
