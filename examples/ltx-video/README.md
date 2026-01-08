# candle-ltx-video: Text-to-Video Generation with DiT

LTX-Video is a powerful text-to-video generation model developed by Lightricks, using a Diffusion Transformer (DiT) architecture for high-quality video synthesis.
[HuggingFace Model](https://huggingface.co/Lightricks/LTX-Video).

## Model Architecture

- **Transformer**: Diffusion Transformer (DiT) optimized for video generation.
- **Text Encoder**: T5-XXL (supports GGUF quantization for memory efficiency).
- **VAE**: 3D AutoEncoder for spatial and temporal video compression.
- **Scheduler**: Flow Match Euler Discrete Scheduler.

## Running the Model

### Basic Usage (Auto-download from HuggingFace)

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --prompt "A serene mountain lake at sunset, photorealistic, 4k" \
    --width 768 --height 512 --num-frames 97 \
    --steps 30
```

### Using Local Weights

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A cute robot dancing in a neon city" \
    --width 384 --height 256 --num-frames 25 \
    --steps 20
```

### Low VRAM Mode (VAE Tiling)

If you encounter Out of Memory (OOM) errors, enable VAE tiling and slicing:

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --width 512 --height 512 \
    --vae-tiling --vae-slicing
```

## Command-line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--prompt` | The text prompt for video generation | `"A video of..."` |
| `--negative-prompt` | Negative prompt for CFG guidance | (Long default) |
| `--width` | Width of the generated video (divisible by 32) | `768` |
| `--height` | Height of the generated video (divisible by 32) | `512` |
| `--num-frames` | Number of frames (should be 8n + 1) | `97` |
| `--steps` | Number of denoising steps | `30` |
| `--guidance-scale` | Classifier-free guidance scale | `3.0` |
| `--local-weights` | Path to local model weight directory | **Auto-download (HuggingFace)** |
| `--output-dir` | Directory to save results | `"output"` |
| `--vae-tiling` | Enable spatial VAE tiling | `false` |
| `--vae-slicing` | Enable batch VAE slicing | `false` |
| `--gif` | Save output as an animated GIF | `true` |
| `--frames` | Save output as individual PNG frames (exclusive) | `false` |
| `--seed` | Random seed for reproducibility | Random |
| `--cpu` | Run on CPU instead of GPU | `false` |
| `--model-id` | HF model ID (used to download tokenizer if missing) | `"Lightricks/LTX-Video"` |

## Video Size Requirements

- **Spatial**: Dimensions must be divisible by 32 (due to 32x VAE compression).
- **Temporal**: Number of frames should ideally follow `8n + 1` (e.g., 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97).

## Performance Notes

- **Flash Attention**: Highly recommended for NVIDIA GPUs to reduce memory usage and increase speed.
- **BF16 Inference**: The model runs in Brain Float 16 by default for optimal performance.
- **Memory**: 97 frames at 512x768 requires ~24GB VRAM without optimizations, ~16GB with VAE tiling.

## Example Outputs

### With VAE Tiling
![VAE Tiling Example](video_with_tiling2.gif)

### Without Tiling
![Standard Example](video_without_tiling.gif)

## Technical Details

### Latent Space

The VAE uses a 32x spatial compression and 8x temporal compression.
- `latent_height = height / 32`
- `latent_width = width / 32`
- `latent_frames = (num_frames - 1) / 8 + 1`

### Position Encoding
LTX-Video uses sophisticated position embeddings to handle temporal and spatial dimensions within the transformer blocks.

### Flow Matching
The model is trained using Flow Matching, which typically allows for high-quality generation in fewer steps (20-30) compared to traditional diffusion.
