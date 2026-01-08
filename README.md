# candle-video

[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/FerrisMind/candle-video/blob/main/LICENSE)

**candle-video** is a Rust library for video generation using AI models, built on top of the [Candle](https://github.com/huggingface/candle) ML framework. It provides high-performance inference for state-of-the-art video generation models.

## Demonstration

**Prompt:** *The waves crash against the jagged rocks of the shoreline, sending spray high into the air. The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.*

![Waves and Rocks](file:///c:/candle-video/examples/ltx-video/video_without_tiling.gif)

🌐 **[Русская версия (Russian)](README.RU.md)**

## Supported Models

- **[LTX-Video](https://huggingface.co/Lightricks/LTX-Video)** — Text-to-video generation using DiT (Diffusion Transformer) architecture
  - Transformer-based diffusion model
  - T5-XXL text encoder (with GGUF quantization support)
  - 3D VAE for video encoding/decoding
  - Flow Matching scheduler

- **Stable Video Diffusion (SVD)** — Image-to-video generation
  - UNet-based architecture
  - CLIP image encoder
  - Temporal VAE
  - EulerA scheduler

## Features

- 🚀 **High Performance** — Native Rust with GPU acceleration via CUDA/cuDNN
- 💾 **Memory Efficient** — BF16 inference, VAE tiling/slicing, GGUF quantized text encoders
- 🔧 **Flexible** — Run on CPU or GPU, with optional Flash Attention
- 📦 **Standalone** — No Python runtime required in production

### Hardware Acceleration

| Feature | Description |
|---------|-------------|
| `cuda` | CUDA backend for NVIDIA GPUs |
| `cudnn` | cuDNN for faster convolutions |
| `flash-attn` | Flash Attention v2 for efficient attention |
| `mkl` | Intel MKL for optimized CPU operations (x86_64) |
| `accelerate` | Apple Accelerate for Metal (macOS) |
| `nccl` | Multi-GPU support via NCCL |

## Installation

### Prerequisites

- Rust 1.82+ (edition 2024)
- CUDA Toolkit 12.x (for GPU acceleration)
- cuDNN 8.x/9.x (optional, for faster convolutions)

### Add to your project

```toml
[dependencies]
candle-video = { git = "https://github.com/FerrisMind/candle-video" }
```

### Build with GPU support

```bash
# Default build (CUDA + cuDNN + Flash Attention)
cargo build --release

# CPU-only build
cargo build --release --no-default-features

# With specific features
cargo build --release --features "cuda,flash-attn"
```

### Windows-specific Notes

For Windows users with CUDA, run the following before building:

```powershell
# Set up environment
.\build_env.ps1

# Build with Flash Attention (optional)
.\build_with_flash_attn.cmd
```

## Quick Start

### LTX-Video: Text-to-Video Generation

#### 1. Automatic usage (Recommended)
Weights will be automatically downloaded from [oxide-lab/LTX-Video-0.9.5](https://huggingface.co/oxide-lab/LTX-Video-0.9.5).

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --prompt "A serene mountain lake at sunset, photorealistic, 4k" \
    --width 768 --height 512 --num-frames 97 \
    --steps 30
```

#### 2. Manual usage (Local weights)
If you already have weights, provide the path:

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A cat playing with a ball of yarn" \
    --vae-tiling
```
# Fast preview (384x256, 25 frames)
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A futuristic cityscape with flying cars" \
    --height 256 \
    --width 384 \
    --num-frames 25 \
    --steps 20

# Low VRAM mode (with VAE tiling)
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A majestic eagle soaring over snow-capped mountains" \
    --vae-tiling --vae-slicing
```

See [examples/ltx-video](examples/ltx-video/README.md) for more details.

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | "A video of a cute cat..." | Text prompt for generation |
| `--negative-prompt` | "low quality, worst quality..." | Negative prompt |
| `--height` | 512 | Video height (must be divisible by 32) |
| `--width` | 768 | Video width (must be divisible by 32) |
| `--num-frames` | 97 | Number of frames (should be 8n + 1) |
| `--steps` | 30 | Diffusion steps |
| `--guidance-scale` | 3.0 | Classifier-free guidance scale |
| `--local-weights` | (None) | Path to local model weights (required) |
| `--output-dir` | "output" | Directory to save results |
| `--seed` | random | Random seed for reproducibility |
| `--vae-tiling` | false | Enable VAE tiling for memory efficiency |
| `--vae-slicing` | false | Enable VAE batch slicing |
| `--frames` | false | Save individual PNG frames (disables GIF) |
| `--gif` | true | Save as animated GIF (default) |
| `--cpu` | false | Run on CPU instead of GPU |
| `--model-id` | "Lightricks/LTX-Video" | HF model ID (for tokenizer download) |

### Library Usage

```rust
use candle_core::{Device, DType};
use candle_video::models::ltx_video::{
    LtxVideoTransformer3DModel,
    AutoencoderKLLtxVideo,
    FlowMatchEulerDiscreteScheduler,
    loader::WeightLoader,
};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    
    // Load Transformer
    let loader = WeightLoader::new(device.clone(), dtype);
    let vb = loader.load_single("path/to/transformer.safetensors")?;
    let config = LtxVideoTransformer3DModelConfig::default();
    let transformer = LtxVideoTransformer3DModel::new(&config, vb)?;
    
    // Load VAE
    let vae_vb = loader.load_single("path/to/vae.safetensors")?;
    let mut vae = AutoencoderKLLtxVideo::new(
        AutoencoderKLLtxVideoConfig::default(),
        vae_vb
    )?;
    
    // Enable memory optimizations
    vae.use_tiling = true;
    vae.use_slicing = true;
    
    // ... setup pipeline and generate
    Ok(())
}
```

## Project Structure

```
candle-video/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── models/
│   │   ├── ltx_video/      # LTX-Video model components
│   │   │   ├── ltx_transformer.rs    # DiT transformer
│   │   │   ├── vae.rs                # 3D VAE
│   │   │   ├── text_encoder.rs       # T5 text encoder
│   │   │   ├── quantized_t5_encoder.rs # GGUF T5 encoder
│   │   │   ├── scheduler.rs          # Flow matching scheduler
│   │   │   ├── t2v_pipeline.rs       # Text-to-video pipeline
│   │   │   └── loader.rs             # Weight loading utilities
│   │   └── svd/            # Stable Video Diffusion components
│   │       ├── unet/       # UNet architecture
│   │       ├── vae/        # Temporal VAE
│   │       ├── clip.rs     # CLIP image encoder
│   │       ├── pipeline.rs # Generation pipeline
│   │       └── scheduler.rs# EulerA scheduler
│   └── utils/              # Utilities
├── examples/               # Usage examples (run with --example)
│   ├── ltx_video/          # Text-to-video example
│   │   ├── main.rs         # Entry point
│   │   └── README.md       # Detailed guide
│   └── verify/             # Verification and debug tools
├── scripts/                # Python verification scripts
├── tests/                  # Integration tests
├── prebuilt/               # Prebuilt kernels (optional)
└── tp/                     # Third-party submodules
```

## Model Weights

### LTX-Video

Download from [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video):

```bash
# Using huggingface-cli
huggingface-cli download Lightricks/LTX-Video --local-dir ./models/ltx-video

# For GGUF T5 encoder (memory efficient)
# Download t5-v1_1-xxl-encoder-Q5_K_M.gguf
```

**Required weight files:**
- `transformer/diffusion_pytorch_model.safetensors` — DiT model
- `vae/diffusion_pytorch_model.safetensors` — 3D VAE
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — Quantized T5
- `tokenizer/tokenizer.json` — T5 tokenizer

## Memory Optimization

For limited VRAM, enable these options:

```bash
# VAE tiling - processes image in tiles
--vae-tiling

# VAE slicing - processes batches sequentially
--vae-slicing

# Lower resolution
--height 256 --width 384

# Fewer frames
--num-frames 25
```

**Approximate VRAM requirements (512x768, 97 frames):**
- Full model: ~24GB
- With VAE tiling: ~16GB
- With GGUF T5: saves ~8GB

## Comparison with PyTorch/diffusers

| Feature | candle-video | diffusers (Python) |
|---------|-------------|-------------------|
| Runtime | Rust native | Python + PyTorch |
| Startup | ~2 seconds | ~15-30 seconds |
| Binary size | ~50MB | ~2GB+ (with deps) |
| VRAM usage | Optimized | Standard |
| Deployment | Single binary | Python environment |

## Common Issues

### CUDA not found

```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### cuDNN errors on Windows

Copy and rename these DLLs to PATH:
- `nvcuda.dll` → `cuda.dll`
- `cublas64_12.dll` → `cublas.dll`
- `curand64_10.dll` → `curand.dll`

### Out of Memory

Try reducing resolution, frames, or enabling VAE tiling:
```bash
--height 256 --width 384 --num-frames 25 --vae-tiling
```

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) — Minimalist ML framework for Rust
- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — Original LTX-Video model
- [Stability AI](https://stability.ai/) — Stable Video Diffusion
- [diffusers](https://github.com/huggingface/diffusers) — Reference implementation
