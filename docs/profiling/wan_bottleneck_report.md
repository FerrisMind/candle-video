# Wan 2.1 bottleneck report (RTX 3060 12 GB)

## Baseline (pre-optimization)

| Metric | Value | Tool |
|--------|-------|------|
| GPU | RTX 3060 12 GB | nvidia-smi |
| Root OOM cause | Missing `flash-attn` → O(n²) attention at ~32k tokens | code audit |
| Transformer default dtype | F32 (~5.2 GB weights) | video-generate |
| Text encoder | UMT5-XXL on CPU | device plan |
| VAE | CPU during denoise | device plan |

## Bottleneck classification

| ID | Class | Evidence | Fix |
|----|-------|----------|-----|
| B1 | GPU memory — excessive intermediate buffers | Attention `[B,H,Q,K]` without flash | Require `flash-attn` on CUDA; runtime guard |
| B2 | dtype conversion overhead | F32 latents + per-block F32 norms | Default F16 transformer; latents in transformer dtype |
| B3 | unnecessary host↔device transfer | VAE on GPU during denoise | Keep VAE on CPU (LTX pattern) |
| B4 | CPU dispatch overhead | CFG = 2× transformer forwards | Document; optional guidance=1 for bench |
| B5 | buffer allocation churn | Per-step `to_dtype` copies | Latents stay F32 for scheduler; single cast per step |

## Optimizations applied

1. `require_flash_attn_for_cuda` / `ensure_wan_cuda_requirements` — fail fast without flash-attn
2. `WanDevicePlan` — shared placement (F16 transformer, TE/VAE on CPU)
3. Feature-gated `profile_zone!` spans on denoise/transformer/VAE
4. `wan-profile` fixtures + `wan_benchmarks` + `scripts/profiling/run_wan_profile.sh`
5. CLI warns when built with `cuda` but without `flash-attn`
6. UMT5 text encoder loads **F32 on CPU** (decoupled from transformer dtype)

## Before / after (RTX 3060 12 GB, `flash-attn,cuda`, transformer F16)

| Scenario | Before (no flash-attn / F32) | After | Notes |
|----------|------------------------------|-------|-------|
| model-load VRAM peak | OOM or >11 GB | **7885 MiB** | `wan-profile model-load` |
| prefill forward (256×448×9) | OOM | **8339 MiB peak**, 2.05s | 1344 tokens |
| decode 2 steps (480×832×9) | OOM | **4792 MiB peak**, 162s | 4680 tokens, CFG 5.0 |
| e2e video-generate (256×448×9×2) | dtype/OOM errors | **OK**, ~2.8 min | guidance 5.0 |
| VRAM after run | — | **~920 MiB** | freed after completion |

## Root cause

**B1 confirmed**: Building with `--features cuda` only (no `flash-attn`) materializes attention `[B,12,Q,K]` in F16. At 480×832×81 → Q≈32760 → **~26 GB** per self-attention layer peak → CUDA OOM on 12 GB.

This was misreported as "model too large" — the 1.3B transformer in F16 is ~2.6 GB weights + ~2–5 GB activations with flash-attn.

## Commands

See `docs/profiling/wan.md`.
