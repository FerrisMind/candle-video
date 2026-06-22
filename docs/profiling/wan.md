# Wan 2.1 profiling & performance

Fixed inputs live under `tests/fixtures/wan/`. Set:

```bash
export WAN_WEIGHTS=/path/to/Wan2.1-T2V-1.3B-Diffusers
```

## Build

```bash
# Production inference (required on 12 GB CUDA)
cargo build --release --example video-generate --features flash-attn,cuda

# Profiling build
cargo build --release --example wan-profile --features flash-attn,cuda,profiling

# Tracy
cargo build --release --example wan-profile --features flash-attn,cuda,profiling,tracy
```

## Profiling fixtures (`wan-profile`)

| Scenario | Purpose | Example |
|----------|---------|---------|
| `synthetic-hot-op` | CPU/GPU dispatch overhead | `--seq-len 512 --iters 200` |
| `model-load` | Weight upload + shader compile | |
| `prefill` | One transformer forward | `--height 256 --width 448 --num-frames 9` |
| `decode` | Denoise loop (+ VAE) | `--steps 5` |
| `full-inference` | Short e2e minus text encoder | `--steps 3` |

```bash
cargo run --release --example wan-profile --features flash-attn,cuda,profiling -- \
  --weights "$WAN_WEIGHTS" prefill --height 256 --width 448 --num-frames 9
```

## Tool wrappers

```bash
chmod +x scripts/profiling/run_wan_profile.sh

# Baseline timing + VRAM
./scripts/profiling/run_wan_profile.sh baseline prefill --height 256 --width 448 --num-frames 9

# Flamegraph
./scripts/profiling/run_wan_profile.sh flamegraph prefill --height 256 --width 448 --num-frames 9

# Heap allocations
./scripts/profiling/run_wan_profile.sh heaptrack model-load

# Tracy (start Tracy GUI first)
./scripts/profiling/run_wan_profile.sh tracy decode --steps 3

# perf/sys-prof
./scripts/profiling/run_wan_profile.sh perf prefill --height 256 --width 448 --num-frames 9
```

## Criterion benchmarks

```bash
cargo bench --features flash-attn,cuda --bench wan_benchmarks
```

## Known 12 GB requirements

1. Build with `--features flash-attn,cuda` (materialized attention OOMs at ~32k tokens).
2. Transformer defaults to **F16** on CUDA (`WanDevicePlan`).
3. Text encoder + VAE stay on CPU during denoise (shared with LTX `MemoryOptions` pattern).

## Profiling loop

1. `cargo bench --bench wan_benchmarks` → baseline
2. `./scripts/profiling/run_wan_profile.sh` → profile
3. Update `docs/profiling/wan_bottleneck_report.md`
4. Apply optimization
5. Re-run bench + profile
6. `cargo test --features flash-attn,cuda` → correctness
