//! Shared device / dtype placement for video backends (Wan, LTX CLI).

use candle_core::{DType, Device, Result};

use super::memory::MemoryOptions;

/// Resolved placement for a Wan inference run on a given GPU budget.
#[derive(Debug, Clone)]
pub struct WanDevicePlan {
    pub compute: Device,
    pub text_encoder: Device,
    pub vae: Device,
    pub transformer_dtype: DType,
    pub vae_dtype: DType,
    /// Spatial VAE tiling (fallback for OOM; default off — decode_full after transformer drop fits 12 GB).
    pub vae_tiling: bool,
    /// TE encode on CPU, then drop before transformer (LTX-style time sequencing).
    pub sequential_gpu: bool,
}

impl WanDevicePlan {
    /// Build a plan for RTX 3060-class 12 GB GPUs.
    ///
    /// Defaults: text encoder is sequenced before the transformer, VAE is
    /// deferred until decode, transformer F16 on CUDA, flash-attn required.
    pub fn for_wan(
        cpu: bool,
        transformer_f16: Option<bool>,
        memory: &MemoryOptions,
    ) -> Result<Self> {
        if cpu {
            return Ok(Self {
                compute: Device::Cpu,
                text_encoder: Device::Cpu,
                vae: Device::Cpu,
                transformer_dtype: DType::F32,
                vae_dtype: DType::F32,
                vae_tiling: false,
                sequential_gpu: false,
            });
        }

        let compute = Device::new_cuda(0)?;
        crate::profiling::vram::log_vram("device_plan_start");

        let text_encoder = if memory.cpu_offload {
            Device::Cpu
        } else {
            // UMT5-XXL (~11 GB bf16) cannot share 12 GB with transformer.
            Device::Cpu
        };

        // LTX-style sequencing: TE encodes first then drops. With CPU
        // offload, VAE is deferred as well and loaded only after denoising.
        let sequential_gpu = !cpu && text_encoder.is_cpu();
        // A 12 GB GPU needs tiled decode for the full 832x480x81 preset. The
        // low-VRAM switch therefore opts into the existing VAE tile path in
        // addition to deferring the VAE load; this changes only execution
        // order/working-set size, not resolution, frame count, or denoise.
        let vae_tiling = memory.vae_tiling || memory.cpu_offload;

        let vae = if memory.cpu_offload || memory.vae_tiling {
            // Keep VAE off the GPU while the transformer is resident. It is
            // loaded onto the compute device only after denoising drops the
            // transformer, avoiding the decode-time VRAM peak on 12 GB cards.
            Device::Cpu
        } else {
            compute.clone()
        };

        let transformer_dtype = match transformer_f16 {
            Some(true) => DType::F16,
            Some(false) => DType::F32,
            None => DType::F16,
        };

        let vae_dtype = wan_vae_baseline_dtype();

        Ok(Self {
            compute,
            text_encoder,
            vae,
            transformer_dtype,
            vae_dtype,
            vae_tiling,
            sequential_gpu,
        })
    }
}

/// Wan 2.1 VAE baseline dtype. The local checkpoint and Diffusers reference use FP32;
/// lower precision is an explicit future parity profile, never an implicit fallback.
fn wan_vae_baseline_dtype() -> DType {
    DType::F32
}

/// Runtime checks for CUDA Wan inference.
pub fn ensure_wan_cuda_requirements(device: &Device, seq_len: usize) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        if seq_len > 4096 {
            candle_core::bail!(
                "Wan CUDA inference requires `--features flash-attn,cuda` for seq_len={seq_len} \
                 (materialized attention would need ~{} GB). Rebuild with flash-attn enabled.",
                (seq_len as u64 * seq_len as u64 * 12 * 2) / (1024 * 1024 * 1024)
            );
        }
    }

    #[cfg(feature = "flash-attn")]
    {
        let _ = seq_len;
    }

    Ok(())
}

/// Patch-token count for Wan latents at a given resolution.
pub fn wan_patch_token_count(height: usize, width: usize, num_frames: usize) -> usize {
    let temporal = 4usize;
    let spatial = 8usize;
    let num_latent_frames = num_frames.saturating_sub(1) / temporal + 1;
    let lh = height / spatial;
    let lw = width / spatial;
    let p_t = 1usize;
    let p_h = 2usize;
    let p_w = 2usize;
    (num_latent_frames / p_t) * (lh / p_h) * (lw / p_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wan_vae_baseline_uses_f32() {
        assert_eq!(wan_vae_baseline_dtype(), DType::F32);
    }
}
