//! VRAM snapshot helpers for profiling reports.

use std::process::Command;

/// Best-effort GPU memory snapshot via `nvidia-smi`.
pub fn snapshot_nvidia_smi() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Log VRAM snapshot when profiling feature is enabled.
pub fn log_vram(label: &str) {
    if let Some(s) = snapshot_nvidia_smi() {
        tracing::info!(target: "candle_video::profiling", label, vram = %s);
    }
}
