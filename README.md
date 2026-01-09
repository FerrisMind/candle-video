</p>
<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

# candle-video

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange)](https://www.rust-lang.org/)

Rust library for AI video generation built on the [Candle](https://github.com/huggingface/candle) ML framework. High-performance, standalone video generation inference without Python runtime dependencies.

---

## 📚 Table of Contents

- [What is this?](#-what-is-this)
- [Key Features](#-key-features)
- [Demonstration](#-demonstration)
- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
- [How to Start Using](#-how-to-start-using)
- [CLI Options](#cli-options)
- [Supported Model Versions](#supported-model-versions)
- [Memory Optimization](#memory-optimization)
- [Project Structure](#project-structure)
- [Acknowledgments](#-acknowledgments)
- [License](#license)

---

## ✨ What is this?

**candle-video** is a Rust-native implementation of video generation models, targeting deployment scenarios where startup time, binary size, and memory efficiency matter. It provides inference for state-of-the-art text-to-video models without requiring a Python runtime.

### Supported Models

- **[LTX-Video](https://huggingface.co/Lightricks/LTX-Video)** — Text-to-video generation using DiT (Diffusion Transformer) architecture
  - 2B and 13B parameter variants
  - Standard and distilled versions (0.9.5 – 0.9.8)
  - T5-XXL text encoder with GGUF quantization support
  - 3D VAE for video encoding/decoding
  - Flow Matching scheduler

---

## 🚀 Key Features

- **High Performance** — Native Rust with GPU acceleration via CUDA/cuDNN
- **Memory Efficient** — BF16 inference, VAE tiling/slicing, GGUF quantized text encoders
- **Flexible** — Run on CPU or GPU, with optional Flash Attention v2
- **Standalone** — No Python runtime required in production
- **Fast Startup** — ~2 seconds vs ~15-30 seconds for Python/PyTorch

### Hardware Acceleration

| Feature | Description |
|---------|-------------|
| `flash-attn` | Flash Attention v2 for efficient attention (default) |
| `cudnn` | cuDNN for faster convolutions (default) |
| `mkl` | Intel MKL for optimized CPU operations (x86_64) |
| `accelerate` | Apple Accelerate for Metal (macOS) |
| `nccl` | Multi-GPU support via NCCL |

---

## 🎬 Demonstration

| Model | Video | Prompt |
| :--- | :---: | :--- |
| **LTX-Video-0.9.5** | ![Waves and Rocks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/Waves_and_Rocks.gif) | *The waves crash against the jagged rocks of the shoreline, sending spray high into the air...* |
| **LTX-Video-0.9.8-2b-distilled** | ![woman_with_blood](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/woman_with_blood.gif) | *A woman with blood on her face and a white tank top looks down and to her right...* |

More examples in [examples](examples/).

---

## 🖥️ System Requirements

### Prerequisites

- [**Rust**](https://rust-lang.org/learn/get-started/) 1.82+ (Edition 2024)
- [**CUDA Toolkit**](https://developer.nvidia.com/cuda-12-6-0-download-archive) 12.x (for GPU acceleration)
- [**cuDNN**](https://developer.nvidia.com/cudnn) 8.x/9.x (optional, for faster convolutions)
- [**hf**](https://huggingface.co/docs/huggingface_hub/guides/cli)

### Approximate VRAM Requirements (512×768, 97 frames)

- Full model: ~8-12GB
- With VAE tiling: ~8GB
- With GGUF T5: saves ~8GB additional

---

## 🛠️ Installation & Setup

### Add to your project

```toml
[dependencies]
candle-video = { git = "https://github.com/FerrisMind/candle-video" }
```

### Build from source

```bash
# Clone the repository
git clone https://github.com/FerrisMind/candle-video.git
cd candle-video

# Default build (CUDA + cuDNN + Flash Attention)
cargo build --release

# CPU-only build
cargo build --release --no-default-features

# With specific features
cargo build --release --features "cudnn,flash-attn"
```

### Model Weights

Download from [oxide-lab/LTX-Video-0.9.8-2B-distilled](https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled):

```bash
huggingface-cli download oxide-lab/LTX-Video-0.9.8-2B-distilled --local-dir ./models/ltx-video
```

> Note: This is the same official version of `Lightricks/LTX-Video` model, , but the repository contains all the necessary files at once. You don't need to individually search for everything

**Required files for diffusers model versions::**
- `transformer/diffusion_pytorch_model.safetensors` — DiT model
- `vae/diffusion_pytorch_model.safetensors` — 3D VAE
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — Quantized T5
- `text_encoder_gguf/tokenizer.json` — T5 tokenizer

**Required files for official model versions:**
- ltxv-2b-0.9.8-distilled.safetensors — DiT + 3D VAE in single file
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — Quantized T5
- `text_encoder_gguf/tokenizer.json` — T5 tokenizer

---

## 📖 How to Start Using

### Using Local Weights Examples (Recommended)

**For diffusers model versions:**

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --ltxv-version 0.9.5 \
    --prompt "A cat playing with a ball of yarn" 
```

**For official model versions:**

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video-model \
    --unified-weights ./models/ltx-video-model.safetensors \
    --ltxv-version 0.9.8-2b-distilled \
    --prompt "A cat playing with a ball of yarn" 
```

### Fast Preview (Lower Resolution)

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video-model \
    --unified-weights ./models/ltx-video-model.safetensors \
    --ltxv-version 0.9.8-2b-distilled \
    --prompt "A cat playing with a ball of yarn" \
    --height 256 --width 384 --num-frames 25 
```

### Low VRAM Mode

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A majestic eagle soaring over mountains" \
    --vae-tiling --vae-slicing
```

---

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | "A video of a cute cat..." | Text prompt for generation |
| `--negative-prompt` | "" | Negative prompt |
| `--height` | 512 | Video height (divisible by 32) |
| `--width` | 768 | Video width (divisible by 32) |
| `--num-frames` | 97 | Number of frames (should be 8n + 1) |
| `--steps` | (from version config) | Diffusion steps |
| `--guidance-scale` | (from version config) | Classifier-free guidance scale |
| `--ltxv-version` | "0.9.5" | Model version |
| `--local-weights` | (None) | Path to local weights |
| `--output-dir` | "output" | Directory to save results |
| `--seed` | random | Random seed for reproducibility |
| `--vae-tiling` | false | Enable VAE tiling for memory efficiency |
| `--vae-slicing` | false | Enable VAE batch slicing |
| `--frames` | false | Save individual PNG frames |
| `--gif` | false | Save as GIF animation |
| `--cpu` | false | Run on CPU instead of GPU |
| `--use-bf16-t5` | false | Use BF16 T5 instead of GGUF quantized |
| `--unified-weights` | (None) | Path to unified safetensors file |

---

## Supported Model Versions

| Version | Parameters | Steps | Guidance | Notes |
|---------|------------|-------|----------|-------|
| `0.9.5` | 2B | 40 | 3.0 | Standard model |
| `0.9.6-dev` | 2B | 40 | 3.0 | Development version |
| `0.9.6-distilled` | 2B | 8 | 1.0 | Fast inference |
| `0.9.8-2b-distilled` | 2B | 7 | 1.0 | Latest distilled |
| `0.9.8-13b-dev` | 13B | 30 | 8.0 | Large model |
| `0.9.8-13b-distilled` | 13B | 7 | 1.0 | Large distilled |

---

## Memory Optimization

For limited VRAM:

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

---

## Project Structure

```
candle-video/
├── src/
│   ├── lib.rs                    # Library entry point
│   └── models/
│       └── ltx_video/            # LTX-Video implementation
│           ├── ltx_transformer.rs    # DiT transformer
│           ├── vae.rs                # 3D VAE
│           ├── text_encoder.rs       # T5 text encoder
│           ├── quantized_t5_encoder.rs # GGUF T5 encoder
│           ├── scheduler.rs          # Flow matching scheduler
│           ├── t2v_pipeline.rs       # Text-to-video pipeline
│           ├── loader.rs             # Weight loading
│           └── configs.rs            # Model version configs
├── examples/
│   └── ltx-video/                # Main CLI example
├── tests/                        # Parity and unit tests
├── scripts/                      # Python reference generators
└── benches/                      # Performance benchmarks
```

---

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) — Minimalist ML framework for Rust
- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — Original LTX-Video model
- [diffusers](https://github.com/huggingface/diffusers) — Reference implementation

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 FerrisMind
