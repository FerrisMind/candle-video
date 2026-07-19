# Wan2.1 T2V 1.3B

candle-video contains a native Candle implementation of Wan2.1
text-to-video inference. The production path is Rust-only: tokenizer,
UMT5, transformer, scheduler, VAE, RGB conversion, H.264 encoding, and MP4
muxing do not invoke Python, PyTorch, Diffusers, or ffmpeg.

## Checkpoint layout

The loader accepts either:

* the original Diffusers directory (model_index.json, component folders,
  indexed or single SafeTensors), or
* the consolidated Wan bundle (native transformer SafeTensors, VAE
  SafeTensors, GGUF UMT5, and tokenizer JSON).

Run the manifest example after preparing a model:

~~~powershell
cargo run --release --example wan-manifest -- --model-path "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B"
~~~

This writes candle-video-manifest.json beside the weights. The manifest
contains component names, relative file paths, byte sizes, SafeTensors tensor
counts, and dtype histograms. It never copies or rewrites model weights.

The implementation was validated against:

* local Diffusers _diffusers_version: 0.33.0.dev0;
* pinned reference sources in candle-video_refs (Diffusers v0.39.0 and
  official Wan2.1 commit 9737cba9c1c3c4d04b33fcad41c111989865d315);
* Candle 0.10.2.

## Inference

The maintained CLI is available as either video-generate or the
wan21-t2v alias:

~~~powershell
cargo run --release --example wan21-t2v --features cuda,flash-attn -- --weights "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B" --prompt "A cinematic shot of ocean waves hitting dark rocks" --negative-prompt "low quality, blurry, static" --width 832 --height 480 --num-frames 81 --steps 50 --guidance-scale 5 --seed 42 --fps 16 --scheduler-profile official-wan --scheduler-shift 5 --output "output\wan21-test.mp4"
~~~

CPU smoke run:

~~~powershell
cargo run --example wan21-t2v -- --weights "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B" --cpu --width 32 --height 32 --num-frames 5 --steps 2 --prompt "A candle flame" --output "output\wan-smoke.mp4"
~~~

Important options are --negative-prompt, --seed, --fps,
--scheduler-profile, --scheduler-shift, --dtype f16|bf16|f32,
--low-vram, --cpu-offload, --vae-tiling, --frames, and --gif.
--output is an explicit MP4 path; parent directories are created. `--model-path`
is accepted as a visible alias for `--weights`.

The Wan CLI shares the LTX progress regime. Human progress is rendered on
stderr and final paths on stdout; `--progress auto|always|never` controls the
terminal renderer, while `--log-format json` emits one structured JSON event
per line. `--events-jsonl path` mirrors those events to a file. Long text,
transformer, and VAE operations emit heartbeat events (five seconds by
default), and `--gpu-stats` appends best-effort `nvidia-smi` snapshots.
`Ctrl+C` requests cooperative cancellation at safe stage/pass boundaries.

Use `--dump-run-manifest` to write an atomic `run.json` beside the output.
MP4/GIF export and manifests first use a sibling `.partial` file; failed
partials are removed by default and retained only with `--keep-partial`.
The event stages are shared with LTX: input validation, component loading,
prompt encoding, latent preparation, denoising (including CFG passes), VAE
decode, post-processing, video encoding, and finalization.

diffusers is the default profile and reads the local UniPC scheduler
configuration. official-wan uses the official Wan UniPC semantics with an
explicit sample shift (the Wan2.1 reference defaults to 5.0).
flow-match-euler is a deterministic first-order flow-matching profile. Its
floating-point schedule and Euler trajectory are checked against Diffusers
0.33.0 in `tests/wan_scheduler_flow_match_euler_parity.rs`; the fixture records
the exact shifted-extrema behavior of that release. `sample_shift` and
Diffusers `flow_shift` are exposed separately; selecting a profile never
silently translates one into the other.

## Correctness evidence

The repository includes configuration, tokenizer, loader, transformer,
scheduler, UMT5, and VAE fixtures under tests/fixtures/wan. Real local
checkpoint checks performed on the CPU include:

| Component | Result |
| --- | --- |
| Transformer load (825 tensors) | passed |
| Transformer forward fixture | passed |
| VAE decode numerical parity | max abs 1.73e-6, mean abs 2.99e-7, cosine 1.0 |
| Consolidated denoise + decode smoke (32×32, 2 steps) | passed |
| MP4 container smoke | valid ftyp/moov/mdat H.264 file |
| FlowMatch Euler schedule + 4-step trajectory | passed, Diffusers 0.33.0 fixture |
| Native scaled-FP8 UMT5 hidden slice | max diff 3.07e-3 (first 8 values, CPU) |

Run the relevant checks with local weights:

~~~powershell
$env:CANDLE_VIDEO_WAN_DIFFUSERS_ROOT = "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B-Diffusers"
cargo test --test wan_transformer_forward_parity
cargo test --test wan_vae_decode_parity
cargo test --test wan_umt5_hidden_parity
~~~

The scaled-FP8 UMT5 loader dequantizes each FP8 linear weight as
cast(fp8, compute_dtype) * scale_weight. Its acceptance criteria are
explicitly looser than FP32 SafeTensors parity because quantization changes
the numerical result.

## Validation log

The following runs were made on the local Windows host (branch `wan`) with
the real consolidated checkpoint:

* CPU release CLI: `--cpu --dtype f32 --width 32 --height 32 --num-frames 1
  --steps 1 --guidance-scale 1 --seed 42`; prompt encoding 11.36 s,
  denoise+decode 1.27 s, observed working set about 10.3 GB. Output:
  `artifacts/wan-smoke.mp4` (890 bytes).
* CUDA release CLI: RTX 3060 12 GB, `--low-vram --dtype f16`, same dimensions
  and seed; prompt encoding 3.75 s, transformer load 1.42 s,
  denoise+decode 0.36 s. Output: `artifacts/wan-smoke-cuda.mp4` (892 bytes).
* CUDA CFG + FlowMatch smoke: guidance 2, negative prompt, and
  `--scheduler-profile flow-match-euler --scheduler-shift 5`; output
  `artifacts/wan-smoke-cuda-flow-cfg.mp4` (867 bytes).
* CUDA CFG + FlowMatch memory sample: the same 32x32, one-frame, one-step
  low-VRAM run completed in 6.97 s. Sampling `nvidia-smi` under Windows WDDM
  measured 1,278 MiB before launch and 9,732 MiB at peak (8,454 MiB
  incremental); peak process working set was 3,147 MiB. Output:
  `artifacts/wan-smoke-cuda-vram.mp4`. The VRAM value is whole-GPU usage, not
  process-isolated usage, because per-process memory is unavailable under WDDM.
* CUDA + Flash Attention release smoke: first kernel build 22 min 13 s,
  prompt encoding 3.74 s, transformer load 1.45 s, denoise+decode 0.38 s;
  output `artifacts/wan-smoke-cuda-flash.mp4` (890 bytes).
* All MP4 files contain `ftyp`, `moov`, `mdat`, `avc1`, and `avcC` boxes.
  The files are intentionally ignored by Git as generated artifacts.

Build and quality gates:

~~~powershell
cargo fmt --all -- --check
cargo check --workspace
cargo check --workspace --features cuda
cargo check --workspace --features cuda,flash-attn
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features cuda,flash-attn -- -D warnings
cargo build --release --example wan21-t2v
~~~

`--all-features` is not a valid Windows gate in this checkout because the
optional macOS `accelerate` dependency (`accelerate-src`) emits
`library kind 'framework' is only supported on Apple targets`. The CUDA and
default feature sets above are the maximal relevant Windows checks.

## Resource guidance and limitations

The baseline VAE path is FP32. On a 12-GB GPU, use sequential/CPU text
encoding (--low-vram) and build with cuda,flash-attn; materialized
attention without Flash Attention is rejected for large token counts.
VAE tiling is opt-in and is not an implicit OOM fallback.

The default implementation is T2V only. CUDA numerical parity is not claimed
for every mixed-precision kernel; use --dtype f32 for a correctness baseline.
Peak RAM/VRAM and elapsed time depend on resolution, steps, and backend and
should be recorded with examples/wan-profile.

## Production boundary

Python scripts in scripts/wan/reference are differential-test and fixture
tools only. They are never spawned by WanPipeline, WanDenoiseStack, the CLI,
or the video export API.
