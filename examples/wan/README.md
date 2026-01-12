# candle-wan: Wan2.1+ Text-to-Video Generation

Wan2.1 is a powerful text-to-video generation model developed by Alibaba, using a DiT (Diffusion Transformer) architecture for high-quality video synthesis.

## Model Architecture

- **Transformer**: Wan DiT transformer (1.3B parameters)
- **Text Encoder**: UMT5-XXL (GGUF quantized for memory efficiency)
- **VAE**: 3D AutoEncoder with 8x spatial and 4x temporal compression
- **Scheduler**: Flow Match Euler Discrete Scheduler

## Running the Model

### Basic Usage (Local Weights)

```bash
cargo run --example wan --release --features flash-attn,cudnn -- \
    --local-weights ./models/Wan2.1-T2V-1.3B \
    --prompt "A cat walking on the grass"
```

### Custom Video Settings

```bash
cargo run --example wan --release --features flash-attn,cudnn -- \
    --local-weights ./models/Wan2.1-T2V-1.3B \
    --prompt "A serene mountain lake at sunset, photorealistic" \
    --width 1280 --height 720 --num-frames 81 \
    --steps 50 --guidance-scale 5.0 \
    --flow-shift 5.0
```

### 480p Generation (Lower VRAM)

```bash
cargo run --example wan --release --features flash-attn,cudnn -- \
    --local-weights ./models/Wan2.1-T2V-1.3B \
    --prompt "A dog running in the park" \
    --width 832 --height 480 --num-frames 49 \
    --flow-shift 3.0
```

### Save Individual Frames

```bash
cargo run --example wan --release --features flash-attn,cudnn -- \
    --local-weights ./models/Wan2.1-T2V-1.3B \
    --prompt "Ocean waves crashing on rocks" \
    --frames --output-dir my_frames
```

### Reproducible Generation

```bash
cargo run --example wan --release --features flash-attn,cudnn -- \
    --local-weights ./models/Wan2.1-T2V-1.3B \
    --prompt "A butterfly landing on a flower" \
    --seed 42
```

## Command-line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--prompt` | Text prompt for video generation | `"A cat walking on the grass"` |
| `--negative-prompt` | Negative prompt for CFG guidance | `""` (empty) |
| `--width` | Width of the generated video (must be divisible by 16) | `832` |
| `--height` | Height of the generated video (must be divisible by 16) | `480` |
| `--num-frames` | Number of frames to generate | `81` |
| `--steps` | Number of denoising steps | `50` |
| `--guidance-scale` | Classifier-free guidance scale | `5.0` |
| `--local-weights` | Path to local model weights directory | (required) |
| `--output-dir` | Directory to save results | `"output"` |
| `--seed` | Random seed for reproducibility | Random |
| `--cpu` | Run on CPU instead of GPU | `false` |
| `--frames` | Save output as individual PNG frames (disables GIF) | `false` |
| `--gif` | Save as GIF animation (default behavior) | `true` |
| `--flow-shift` | Flow shift for scheduler (5.0 for 720p, 3.0 for 480p) | `5.0` |

## Video Size Requirements

### Spatial Dimensions
- **Width and height must be divisible by 16** (due to VAE compression)
- Common valid sizes: 480, 512, 640, 720, 768, 832, 1024, 1280

### Recommended Resolutions
| Resolution | Width | Height | Use Case |
|------------|-------|--------|----------|
| 480p | 832 | 480 | Lower VRAM, faster generation |
| 720p | 1280 | 720 | Higher quality, more VRAM |
| Square | 512 | 512 | Balanced |

### Temporal Dimensions
- Number of frames can be any positive integer
- Latent frames formula: `(num_frames - 1) / 4 + 1`
- Common frame counts: 17, 33, 49, 65, 81, 97

## Model Weights Directory Structure

The `--local-weights` directory should contain:

```
Wan2.1-T2V-1.3B/
├── wan2.1_t2v_1.3B_fp16.safetensors    # Transformer weights
├── vae/
│   └── wan_2.1_vae.safetensors         # VAE weights
└── text_encoder_gguf/
    ├── umt5-xxl-encoder-Q5_K_M.gguf    # Quantized text encoder
    └── tokenizer.json                   # UMT5 tokenizer
```

## Memory Requirements

### GPU VRAM Estimates

| Resolution | Frames | Estimated VRAM |
|------------|--------|----------------|
| 480x832 | 49 | ~10-12 GB |
| 480x832 | 81 | ~14-16 GB |
| 720x1280 | 49 | ~18-20 GB |
| 720x1280 | 81 | ~24+ GB |

### Memory Optimization Tips

1. **Reduce resolution**: Use 480p instead of 720p
2. **Fewer frames**: Generate 49 frames instead of 81
3. **CPU fallback**: Use `--cpu` flag (much slower but works with limited VRAM)
4. **Quantized text encoder**: GGUF quantization reduces text encoder memory by ~4x

## Technical Details

### Latent Space

The VAE uses 8x spatial compression and 4x temporal compression:
- `latent_height = height / 8`
- `latent_width = width / 8`
- `latent_frames = (num_frames - 1) / 4 + 1`
- `latent_channels = 16`

### Flow Shift Parameter

The `--flow-shift` parameter controls the noise schedule:
- **5.0**: Recommended for 720p resolution
- **3.0**: Recommended for 480p resolution
- Higher values = more aggressive denoising

### Classifier-Free Guidance (CFG)

- `guidance_scale > 1.0`: Enables CFG (computes both conditional and unconditional predictions)
- `guidance_scale <= 1.0`: Disables CFG (faster, single forward pass)
- Recommended range: 3.0 - 7.0

## Output Formats

### GIF Animation (Default)
- Saved as `{output_dir}/video.gif`
- ~25 FPS playback
- Infinite loop

### PNG Frames (--frames flag)
- Saved as `{output_dir}/frame_0000.png`, `frame_0001.png`, etc.
- Lossless quality
- Useful for post-processing or video encoding

## Example Prompts

```bash
# Nature scenes
--prompt "A waterfall cascading down mossy rocks in a lush forest"
--prompt "Cherry blossoms falling in slow motion against a blue sky"

# Animals
--prompt "A golden retriever running through a field of sunflowers"
--prompt "A hummingbird hovering near a red flower"

# Urban scenes
--prompt "City traffic at night with neon lights reflecting on wet streets"
--prompt "A coffee shop window with rain droplets, warm interior lighting"

# Abstract/Artistic
--prompt "Colorful paint swirling in water, abstract patterns forming"
--prompt "Northern lights dancing over a frozen lake"
```

## Troubleshooting

### "Model file not found"
Ensure your model directory structure matches the expected layout above.

### "Invalid dimensions"
Width and height must be divisible by 16. Use values like 480, 512, 640, 720, 768, 832, 1024, 1280.

### Out of Memory (OOM)
- Reduce resolution: `--width 512 --height 512`
- Reduce frames: `--num-frames 33`
- Use CPU: `--cpu` (very slow but works)

### CUDA not available
The example will automatically fall back to CPU if CUDA is not available. For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- Built with `--features flash-attn,cudnn`
