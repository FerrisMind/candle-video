# LTX-Video Performance Benchmarks

## Overview

Performance benchmark results for LTX-Video pipeline components.
Tests were run with features: `flash-attn`, `cudnn`.

## Test Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **OS** | Windows 11 x64 |
| **CPU** | AMD Ryzen 7 3700X @ 4.3 GHz (all cores), 8C/16T |
| **RAM** | 64 GB DDR4 3666 MHz (dual channel) |
| **GPU** | NVIDIA GeForce RTX 3060 12GB VRAM |
| **CUDA** | CUDA Toolkit 12.x |
| **Features** | Flash Attention v2, cuDNN |

## Benchmark Results (Baseline)

### Scheduler Performance (CPU)

| Benchmark | Time | Notes |
|-----------|------|-------|
| step (batch=1, seq=1024, ch=128) | ~4.6 µs | CPU operation |
| step (batch=1, seq=4096, ch=128) | ~4.6 µs | CPU operation |
| step (batch=2, seq=4096, ch=128) | ~4.6 µs | CPU operation |
| set_timesteps (10 steps) | ~4.6 µs | |
| set_timesteps (20 steps) | ~4.9 µs | |
| set_timesteps (40 steps) | ~4.5 µs | |
| set_timesteps (50 steps) | ~4.6 µs | |

**Conclusion**: Scheduler operations are very fast (~5µs), not a bottleneck.

### Transformer Forward (GPU, 4 layers, Flash Attention)

| Configuration | Time | Throughput |
|---------------|------|------------|
| batch=1, seq=1024, frames=9, h=32, w=32 | **10.2 ms** | ~100 fps |
| batch=1, seq=4096, frames=9, h=64, w=64 | **37.6 ms** | ~27 fps |
| batch=1, seq=8192, frames=17, h=64, w=64 | **71.1 ms** | ~14 fps |

**Note**: Tested with 4 layers (instead of 28 in production). 
For full model (28 layers) time will be ~7x higher:
- seq=1024: ~70 ms per step
- seq=4096: ~260 ms per step
- seq=8192: ~500 ms per step

### VAE Decode (GPU)

VAE decode is the most memory-intensive operation.
For production, tiled decoding is recommended.

Typical decode times:
- 256x384, 65 frames: ~10-20 seconds
- 512x768, 97 frames: ~30-60 seconds (with tiling)

### Performance Summary

| Component | % of Total Time | Optimization Priority |
|-----------|-----------------|----------------------|
| Transformer (denoising) | ~70-80% | High (Flash Attention) |
| VAE Decode | ~15-25% | Medium (Tiling) |
| Text Encoder | ~3-5% | Low |
| Scheduler | <1% | None needed |

## Running Benchmarks

```bash
# Full benchmarks (GPU required)
cargo bench --features flash-attn,cudnn

# GPU-only benchmarks (transformer, vae)
cargo bench --features flash-attn,cudnn -- "transformer|vae"

# Scheduler benchmarks only
cargo bench --features flash-attn,cudnn -- scheduler

# CPU-only benchmarks
cargo bench --no-default-features
```

## Performance Recommendations

1. **Transformer**: Main bottleneck. Flash Attention is critical for performance.
2. **VAE Decode**: Use tiled decoding for higher resolutions.
3. **Scheduler**: No optimization needed, already very fast.
4. **Dtype conversions**: Minimize conversions between F32/BF16.

## Comparison Notes

To compare before/after changes:
- Run `cargo bench` and save results
- Criterion automatically compares with previous runs
- Results are saved in `target/criterion/`
- HTML reports: `target/criterion/report/index.html`

## Parity Status

After all optimizations for Python parity:
- ✅ Scheduler: MSE < 1e-6
- ✅ Transformer: MSE < 1e-4
- ✅ VAE: MSE < 1e-3
- ✅ Pipeline: PSNR > 35dB

Performance was not degraded after parity changes.
