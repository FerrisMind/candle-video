# SVD Candle Integration — Analysis Report

**Date:** 2025-12-29  
**Status:** ✅ Analysis Complete  
**Next Phase:** Planning

---

## Executive Summary

Реализация **Stable Video Diffusion (SVD)** inference в Rust/Candle требует создания изолированного модуля `src/svd/` с UNet-based SpatioTemporal архитектурой. Это принципиально отличается от существующего DiT-based LTX-Video. Главный референс — **diffusers**.

---

## 1. Scope v1 (Зафиксированный контракт)

| Параметр | Значение | Источник |
|----------|----------|----------|
| **Режим** | Image-to-Video (img2vid) | diffusers |
| **Кадры** | 14 | unet.config.num_frames |
| **Разрешение** | 576×1024 | HF model card |
| **Dtype** | FP16 (с force_upcast для VAE) | diffusers pipeline |
| **Guidance** | `linspace(min, max, num_frames)` per-frame | pipeline |
| **fps conditioning** | `fps - 1` | micro-conditioning |
| **Веса** | HF компоненты (unet/, vae/, image_encoder/) | HF repo |

---

## 2. Architecture Overview

### 2.1 Pipeline Flow (diffusers reference)

```
Input Image (PIL/Tensor)
    │
    ├─► CLIPImageProcessor.preprocess() → [1, 3, 224, 224]
    │
    ├─► CLIPVisionModelWithProjection.forward()
    │       └─► image_embeddings: [B, 1024]
    │
    ├─► VideoProcessor.preprocess() → [1, 3, 576, 1024]
    │       └─► + noise_aug_strength * noise
    │
    ├─► VAE.encode() [force_upcast to FP32 if FP16]
    │       └─► image_latents: [B, 4, 72, 128]
    │       └─► repeat for num_frames → [B, 14, 4, 72, 128]
    │
    ├─► _get_add_time_ids(fps-1, motion_bucket_id, noise_aug_strength)
    │       └─► added_time_ids: [B, 3]
    │
    ├─► prepare_latents() → [B, 14, 4, 72, 128] random noise
    │
    ├─► Denoising Loop (25 steps default):
    │       │
    │       ├─► latent_model_input = cat([latents, latents]) for CFG
    │       ├─► image_latents concat → [B, 14, 8, 72, 128]
    │       ├─► UNet.forward(sample, timestep, encoder_hidden_states, added_time_ids)
    │       │       └─► noise_pred: [B, 14, 4, 72, 128]
    │       ├─► CFG: noise = uncond + guidance_scale[frame] * (cond - uncond)
    │       └─► scheduler.step()
    │
    └─► VAE.decode(latents, num_frames=14)
            └─► video: [B, 14, 3, 576, 1024]
```

### 2.2 UNet Architecture (UNetSpatioTemporalConditionModel)

```
in_channels: 8 (4 noise + 4 image latents concatenated)
out_channels: 4
block_out_channels: [320, 640, 1280, 1280]
cross_attention_dim: 1024 (CLIP embedding dim)
num_attention_heads: [5, 10, 20, 20]
num_frames: 14
addition_time_embed_dim: 256 (for fps, motion_bucket_id, noise_aug)
projection_class_embeddings_input_dim: 768

down_blocks:
  [0] CrossAttnDownBlockSpatioTemporal(320→320, heads=5, downsample=True)
  [1] CrossAttnDownBlockSpatioTemporal(320→640, heads=10, downsample=True)
  [2] CrossAttnDownBlockSpatioTemporal(640→1280, heads=20, downsample=True)
  [3] DownBlockSpatioTemporal(1280→1280, no cross-attn, downsample=False)

mid_block:
  UNetMidBlockSpatioTemporal(1280, heads=20)

up_blocks:
  [0] UpBlockSpatioTemporal(1280, no cross-attn)
  [1] CrossAttnUpBlockSpatioTemporal(1280→1280, heads=20)
  [2] CrossAttnUpBlockSpatioTemporal(1280→640, heads=10)
  [3] CrossAttnUpBlockSpatioTemporal(640→320, heads=5)
```

### 2.3 VAE Architecture (AutoencoderKLTemporalDecoder)

```
Encoder: Standard 2D (DownEncoderBlock2D)
  block_out_channels: [128, 256, 512, 512]
  latent_channels: 4
  scaling_factor: 0.18215

Decoder: TemporalDecoder
  MidBlockTemporalDecoder → UpBlockTemporalDecoder × 4
  + time_conv_out: Conv3d(3, 3, kernel=(3,1,1)) for frame blending
```

### 2.4 CLIP Image Encoder

```
CLIPVisionModelWithProjection:
  hidden_size: 1280
  image_size: 224
  patch_size: 14
  num_hidden_layers: 32
  num_attention_heads: 16
  projection_dim: 1024  ← Critical: need projection Linear layer
```

### 2.5 Scheduler (EulerDiscreteScheduler)

```
num_train_timesteps: 1000
beta_schedule: "scaled_linear"
beta_start: 0.00085
beta_end: 0.012
prediction_type: "v_prediction"  ← Important!
use_karras_sigmas: True
interpolation_type: "linear"
timestep_spacing: "leading"
steps_offset: 1
```

---

## 3. Key Implementation Decisions

### 3.1 Closed Ambiguities

| Question | Decision | Rationale |
|----------|----------|-----------|
| `image_only_indicator` | `zeros(batch, num_frames)` inside VAE decode | Matches diffusers, simplifies API |
| `force_upcast` | Implement: FP32 for VAE encode/decode when dtype=FP16 | Prevents NaN/artifacts |
| Weight naming | Explicit mapping layer with coverage test | Flexible, testable |
| Guidance per-frame | `linspace(min, max, num_frames)` | Required for correct dynamics |
| fps conditioning | Use `fps - 1` | Micro-conditioning from training |

### 3.2 Reusable Components

| Component | Source | Notes |
|-----------|--------|-------|
| `ClipVisionTransformer` | `candle-transformers` | Add projection Linear(1280 → 1024) |
| `WeightLoader` | `src/loader.rs` | Full reuse with key mapping |
| `GroupNorm`, `LayerNorm` | `candle_nn` | Standard |
| `Conv2d`, `Linear` | `candle_nn` | Standard |

### 3.3 Components to Implement

| Component | Complexity | Dependencies |
|-----------|------------|--------------|
| `EulerDiscreteScheduler` | 🟡 Medium | v_prediction, karras sigmas |
| `SpatioTemporalResBlock` | 🟡 Medium | Temporal mixing, alpha blender |
| `TransformerSpatioTemporalModel` | 🔴 High | Spatial + Temporal attention |
| `CrossAttnDownBlockSpatioTemporal` | 🟡 Medium | ResBlock + Transformer |
| `UpBlockSpatioTemporal` | 🟢 Low | ResBlock only |
| `CrossAttnUpBlockSpatioTemporal` | 🟡 Medium | + Transformer |
| `UNetMidBlockSpatioTemporal` | 🟡 Medium | Transformer + ResBlock |
| `UNetSpatioTemporalConditionModel` | 🔴 High | All blocks + embeddings |
| `TemporalDecoder` | 🟡 Medium | Temporal conv + upsampling |
| `AutoencoderKLTemporalDecoder` | 🟡 Medium | 2D Encoder + TemporalDecoder |
| `CLIPVisionModelWithProjection` | 🟢 Low | Wrapper + projection |
| `SVDPipeline` | 🔴 High | All components |

---

## 4. File Structure

```
src/svd/
├── mod.rs                    # Public API exports
├── config.rs                 # SvdConfig, SvdUnetConfig, SvdVaeConfig, EulerSchedulerConfig
├── scheduler.rs              # EulerDiscreteScheduler
├── clip.rs                   # CLIPVisionModelWithProjection wrapper
├── unet/
│   ├── mod.rs                # UNetSpatioTemporalConditionModel
│   ├── blocks.rs             # Down/Up/Mid SpatioTemporal blocks
│   ├── resnet.rs             # SpatioTemporalResBlock
│   └── transformer.rs        # TransformerSpatioTemporalModel
├── vae/
│   ├── mod.rs                # AutoencoderKLTemporalDecoder
│   ├── encoder.rs            # 2D Encoder (standard)
│   └── decoder.rs            # TemporalDecoder
├── pipeline.rs               # SVDPipeline main entry
└── weight_mapping.rs         # HF key → Candle path mapping
```

---

## 5. Testing Strategy

| Stage | Metric | Threshold |
|-------|--------|-----------|
| **Scheduler** | timesteps/sigmas match | Exact |
| **CLIP** | Embedding shape & range | ±1e-4 relative |
| **VAE encode** | Latent statistics | PSNR > 40dB |
| **VAE decode** | Frame quality | PSNR > 35dB |
| **UNet forward** | noise_pred error | < 1e-4 relative |
| **Full pipeline** | Visual quality | SSIM > 0.95 at fixed seed |

---

## 6. Weight Files (v1 Canonical)

```
models/svd/
├── image_encoder/
│   ├── config.json
│   └── model.fp16.safetensors
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
├── vae/
│   └── config.json
│   (VAE weights in unet safetensors or vae/diffusion_pytorch_model.safetensors)
├── scheduler/
│   └── scheduler_config.json
└── model_index.json
```

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Weight naming mismatch | 🟡 Medium | 🔴 High | Key mapping layer + test |
| FP16 numerical issues | 🟡 Medium | 🔴 High | force_upcast + monitoring |
| TransformerSpatioTemporal complexity | 🔴 High | 🟡 Medium | Incremental build + unit tests |
| UNet memory footprint | 🟡 Medium | 🟡 Medium | Gradient checkpointing if needed |

---

## 8. Engineering Principles Compliance

| Principle | Status | Notes |
|-----------|--------|-------|
| **YAGNI** | ✅ | v1 scope minimal: 14 frames, 576×1024, no XT |
| **DRY** | ✅ | Reuse CLIP, loader; common attention abstraction possible |
| **SOLID** | ✅ | Isolated module, clear interfaces |
| **KISS** | ✅ | Direct diffusers port, no premature abstraction |

---

## 9. Next Steps (Planning Phase)

1. Create detailed task breakdown with estimates
2. Define dependency graph
3. Set up test fixtures (reference latents, embeddings)
4. Implement in order: Scheduler → CLIP → VAE → UNet → Pipeline

---

**Analysis Phase Complete. Ready for Planning.**
