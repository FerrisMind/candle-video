//! CUDA build/runtime guards shared across backends.

use candle_core::{DType, Device, Result};

/// Elementwise/norm compute dtype: F16/BF16 on CUDA (matches Diffusers inference), F32 on CPU for parity.
pub fn wan_inference_compute_dtype(device: &Device, storage_dtype: DType) -> DType {
    if device.is_cuda() {
        storage_dtype
    } else {
        DType::F32
    }
}

/// Warn or fail when CUDA is used without flash-attn for large-sequence models.
pub fn require_flash_attn_for_cuda(device: &Device, context: &str, seq_len: usize) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = context;
        if seq_len > 4096 {
            candle_core::bail!(
                "{context}: CUDA build missing `flash-attn` feature (seq_len={seq_len}). \
                 Rebuild: cargo build --release --features flash-attn,cuda"
            );
        }
        tracing::warn!(
            target: "candle_video",
            "{context}: flash-attn disabled; attention falls back to O(n²) materialization"
        );
    }

    #[cfg(feature = "flash-attn")]
    {
        let _ = (context, seq_len);
    }

    Ok(())
}
