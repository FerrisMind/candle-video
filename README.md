# Candle Video

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![RU English](https://img.shields.io/badge/Русский-red)](README.RU.md)
[![EN English](https://img.shields.io/badge/English-blue)](README.md)

Integration of the **LTX-Video** model for the [Candle](https://github.com/huggingface/candle) framework in Rust.

This project implements high-performance inference for the LTX-Video video generation model, achieving faster-than-real-time generation speeds on modern GPUs.

> ⚠️ **Alpha Stage**: This project is currently in **alpha stage**. Correct inference is **not guaranteed**. The API may change, and results may be unstable or incorrect. Use at your own risk.

## 🎯 Features

- **Text-to-Video (T2V)**: Generate videos from text descriptions
- **Image-to-Video (I2V)**: Animate static images
- **High Performance**: Optimized Rust implementation with CUDA support
- **Memory Efficient**: Flash Attention 2 support for efficient handling of large sequences
- **Flexible Data Formats**: Support for F32, BF16, and FP8
- **Rectified Flow**: Modern diffusion scheduler for fast generation

## 📋 Requirements

### System Requirements

- **Rust**: version 1.70 or higher
- **CUDA**: version 11.8 or higher (for GPU acceleration)
- **Git LFS**: for working with large model files
- **Visual Studio Build Tools** (Windows): for compiling CUDA kernels

### Installing Git LFS

```bash
# Install Git LFS (if not already installed)
git lfs install
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/FerrisMind/candle-video.git
cd candle-video
```

### 2. Build the Project

#### Build with Flash Attention (Recommended)

```bash
# Windows
build_flash_attn.cmd

# Linux/macOS
cargo build --release --features flash-attn
```

#### Build without Flash Attention

```bash
cargo build --release
```

### 3. Download the Model

The LTX-Video model should be placed in the `ltxv-2b-0.9.8-distilled/` directory at the project root.

Directory structure:
```
ltxv-2b-0.9.8-distilled/
├── ltxv-2b-0.9.8-distilled.safetensors
├── model_index.json
├── t5-v1_1-xxl-encoder-Q5_K_M.gguf
└── vae/
    └── vae.safetensors
```

### 4. Generate Video

#### Text-to-Video

```bash
cargo run --release --bin ltx-video -- \
    --prompt "A beautiful sunset over the ocean with waves crashing on the shore" \
    --model-path ./ltxv-2b-0.9.8-distilled \
    --output output_video \
    --num-frames 17 \
    --height 512 \
    --width 768 \
    --num-inference-steps 20 \
    --guidance-scale 7.5 \
    --seed 42
```

#### Image-to-Video

```bash
cargo run --release --bin ltx-video -- \
    --mode i2v \
    --prompt "A serene landscape with gentle movement" \
    --image path/to/your/image.jpg \
    --model-path ./ltxv-2b-0.9.8-distilled \
    --output output_video \
    --num-frames 17
```

## 📖 Usage

### CLI Parameters

Main parameters for the `ltx-video` command:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prompt` | Text description of the video | *(required)* |
| `--model-path` | Path to model directory | *(required)* |
| `--output` | Directory to save frames | `output` |
| `--mode` | Generation mode: `t2v` or `i2v` | `t2v` |
| `--image` | Path to image (for i2v) | - |
| `--num-frames` | Number of frames (8N+1: 9, 17, 25...) | `17` |
| `--height` | Video height (multiple of 32) | `512` |
| `--width` | Video width (multiple of 32) | `768` |
| `--num-inference-steps` | Number of inference steps | `20` |
| `--guidance-scale` | Classifier-free guidance scale | `7.5` |
| `--frame-rate` | Frame rate for RoPE | `25.0` |
| `--seed` | Random seed | `42` |
| `--use-flash-attn` | Use Flash Attention | `false` |
| `--f32` | Use FP32 precision | `false` |
| `--fp8` | Use FP8 precision | `false` |
| `--cpu` | Use CPU instead of GPU | `false` |

### Programmatic API

```rust
use candle_video::{
    config::{DitConfig, InferenceConfig, SchedulerConfig, VaeConfig},
    pipeline::{PipelineConfig, TextToVideoPipeline},
    text_encoder::T5TextEncoderWrapper,
};
use candle_core::Device;

// Create configuration
let config = PipelineConfig {
    dit: DitConfig::default(),
    vae: VaeConfig::default(),
    scheduler: SchedulerConfig::default(),
};

// Initialize pipeline
let device = Device::Cpu; // or Device::cuda_if_available(0)?
let pipeline = TextToVideoPipeline::new(&device, config)?;

// Inference configuration
let inference_config = InferenceConfig {
    num_frames: 17,
    height: 512,
    width: 768,
    num_inference_steps: 20,
    guidance_scale: 7.5,
    frame_rate: Some(25.0),
    ..Default::default()
};

// Generate video
let video_frames = pipeline.generate_with_cfg(
    &text_embeddings,
    &inference_config,
    &negative_embeddings,
)?;
```

## 🏗️ Architecture

The project implements a complete video generation pipeline:

```
Text Prompt → T5 Encoder → Text Embeddings
                                    ↓
         Random Noise → DiT Denoising Loop → Denoised Latents
                                    ↓
                           VAE Decoder → Video Frames
```

### Main Components

- **T5 Text Encoder**: Encodes text prompts into embeddings
- **DiT (Diffusion Transformer)**: Transformer for denoising latents
- **Causal Video VAE**: Variational autoencoder with causal 3D convolutions
- **Rectified Flow Scheduler**: Diffusion scheduler for fast generation
- **Fractional RoPE**: Normalized fractional positional encoding

## 🔧 Building and Development

### Build Features

The project uses the following Cargo features:

- `flash-attn`: Flash Attention 2 support (memory efficient)
- `cudnn`: cuDNN acceleration for convolutions
- `mkl`: Intel MKL for CPU acceleration (Linux/Windows x86_64)
- `accelerate`: Apple Accelerate for Metal (macOS)
- `nccl`: Multi-GPU support
- `all-gpu`: All GPU optimizations

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test --test integration

# Run with output
cargo test -- --nocapture
```

### Debugging

Use environment variables for debugging:

```bash
# Enable logging
RUST_LOG=debug cargo run --release --bin ltx-video -- ...
```

## 🐛 Known Limitations

- **Alpha Stage**: The project is in alpha stage. Correct inference is not guaranteed. Results may be unstable or incorrect.
- **Conv3D**: Current implementation uses 3D convolution emulation through 2D operations. Native Conv3D implementation via Custom Op may improve performance.
- **Memory**: High-resolution video generation requires significant GPU memory. It is recommended to use Flash Attention and BF16.

## 📄 License

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Candle](https://github.com/huggingface/candle) - Machine learning framework in Rust
- [Lightricks LTX-Video](https://github.com/Lightricks/LTX-Video) - Original LTX-Video model
- Rust and ML developer community

---

**Note**: This project is in **alpha stage** and under active development. API may change between versions. Correct inference is not guaranteed.

