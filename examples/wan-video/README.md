# candle-wan-video: Wan2.1 Text-to-Video

This example exposes the native Candle implementation of
**Wan2.1-T2V-1.3B** in the same self-contained example layout as
`examples/ltx-video`. It uses the shared `video-generate` implementation, so
the `wan-video` and `wan21-t2v` commands have identical behavior.

## Model architecture

- **Transformer**: WanTransformer3DModel with 3D patch embedding, RoPE,
  self-attention, text cross-attention, Q/K RMSNorm, and unpatchify.
- **Text encoder**: local T5TokenizerFast-compatible tokenizer and UMT5,
  supporting scaled-FP8 SafeTensors and consolidated GGUF.
- **VAE**: causal 3D Wan VAE for latent-to-RGB video decoding.
- **Scheduler**: Diffusers UniPC, official Wan UniPC, or FlowMatch Euler
  compatibility profile.
- **Export**: native PNG, GIF, and H.264/MP4 output; no ffmpeg process.

## Preparing weights

The loader reads either the original Diffusers folder or the consolidated
bundle directly. No Python or PyTorch process is started during inference.

Generate a machine-readable manifest when needed:

```powershell
cargo run --release --example wan-manifest -- --model-path `
  "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B"
```

See [docs/wan21.md](../../docs/wan21.md) for the expected files and tensor
inventory.

## Basic usage

CPU smoke run:

```powershell
cargo run --example wan-video -- --weights `
  "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B" `
  --cpu --width 32 --height 32 --num-frames 5 --steps 2 `
  --prompt "A candle flame" --output "output\wan-smoke.mp4"
```

CUDA generation with low-VRAM phased loading:

```powershell
cargo run --release --example wan-video --features cuda,flash-attn -- `
  --model-path "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B" `
  --prompt "A cinematic shot of ocean waves hitting dark rocks" `
  --negative-prompt "low quality, blurry, static" `
  --width 832 --height 480 --num-frames 81 --steps 50 `
  --guidance-scale 5 --seed 42 --fps 16 --low-vram `
  --scheduler-profile official-wan --scheduler-shift 5 `
  --output "output\wan21-test.mp4"
```

When `--output` is omitted, the default output is `output\video.gif`.
Use `--frames` to save PNG frames instead. `--gif` can be combined with an
explicit MP4 path when both outputs are wanted.

## Progress and reproducibility

Wan uses the same structured progress observer as `examples/ltx-video`.
Human progress is sent to stderr and final output paths to stdout. For a
machine-readable stream and an atomic run manifest:

```powershell
cargo run --release --example wan-video --features cuda,flash-attn -- `
  --model-path "C:\Users\PC\Documents\models\Wan2.1-T2V-1.3B" `
  --prompt "A candle flame" --width 32 --height 32 --num-frames 5 --steps 2 `
  --low-vram --progress always --timings --log-format json `
  --events-jsonl output\wan-events.jsonl --dump-run-manifest `
  --output output\wan-smoke.mp4
```

`Ctrl+C` requests cooperative cancellation at stage/pass boundaries. MP4,
GIF, and the JSON manifest are published through sibling `.partial` files;
failed partial files are removed unless `--keep-partial` is set. Heartbeats
are emitted every five seconds by default, and `--gpu-stats` adds best-effort
`nvidia-smi` snapshots.

## Command-line flags

| Flag | Description | Default |
| --- | --- | --- |
| `--weights` / `--model-path` | Diffusers or consolidated model directory | required |
| `--prompt` | Positive text prompt | `A cat playing with a ball of yarn` |
| `--negative-prompt` | Unconditional prompt for CFG | empty |
| `--width`, `--height` | Requested output dimensions; validated/rounded from Wan config | `832`, `480` |
| `--num-frames` | Requested frame count; normalized to the VAE temporal step | `81` |
| `--steps` | Denoising steps | `50` |
| `--guidance-scale` | Classifier-free guidance scale | `5` |
| `--seed` | Reproducible generation seed | random |
| `--scheduler-profile` | `diffusers`, `official-wan`, or `flow-match-euler` | `diffusers` |
| `--scheduler-shift` | Explicit profile-specific flow/sample shift | config/default |
| `--dtype` | Transformer compute dtype: `f16`, `bf16`, or `f32` | backend default |
| `--fps` | GIF/MP4 frame rate | `16` |
| `--output` | Explicit MP4 path | none |
| `--output-dir` | Directory for GIF/PNG output | `output` |
| `--frames` | Save PNG frames instead of the default GIF | false |
| `--gif` | Also save GIF when an MP4 path is supplied | false |
| `--cpu` | Force CPU execution | false |
| `--low-vram` | Enable sequential text/transformer loading, deferred VAE, and tiled decode | false |
| `--cpu-offload` | Explicit CPU offload path | false |
| `--vae-tiling` | Opt-in VAE tiling | false |
| `--vae-slicing` | Opt-in VAE slicing | false |
| `--transformer-f32` | Force F32 transformer weights | false |
| `--latents-json` | Load fixed initial latents for differential testing | none |
| `--progress` | `auto`, `always`, or `never` terminal rendering | `auto` |
| `--log-format` | `human` or JSONL events on stderr | `human` |
| `--events-jsonl` | Also write events to a JSONL file | none |
| `--heartbeat-seconds` | Interval for long-operation heartbeat events | `5` |
| `--timings` | Include stage elapsed times | false |
| `--diagnostics`, `-v`, `-q` | Diagnostics, verbosity, or quiet human output | false |
| `--gpu-stats` | Add best-effort `nvidia-smi` snapshots | false |
| `--dump-run-manifest` | Write `run.json` beside output | false |
| `--keep-partial` | Keep an incomplete `.partial` output on failure | false |

When `--seed` is omitted, Wan follows the official implementation: it draws a
random integer in `0..=sys.maxsize`, prints the chosen value, and stores it in
`run.json`. Supplying `--seed N` makes the noise reproducible. When
`--latents-json` supplies fixed initial latents, no noise seed is consumed.

## Size and memory notes

Wan validates spatial dimensions against the model patch/VAE requirements
(the normal path uses dimensions divisible by 16) and normalizes frame count
from the configured temporal compression. The recommended preset is
832×480×81. For a 12-GB GPU, build with `cuda,flash-attn` and use
`--low-vram`; the baseline VAE path is FP32.

The measured 32×32 one-frame CUDA smoke used 9,732 MiB peak whole-GPU memory
under Windows WDDM. In `--low-vram` mode the VAE is not loaded at startup: it
is loaded on CUDA only after denoising drops the transformer, then uses the
existing tiled decode path to bound spatial activations. Full-quality
generation time and memory scale with resolution, frame count, steps, and dtype.

## Production boundary

`wan-video` is a Rust/Candle example. Python scripts under `scripts/wan` are
reference and fixture tools only; they are never invoked by the pipeline,
CLI, or native video export path.
