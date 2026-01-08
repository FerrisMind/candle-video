---
base_model: Lightricks/LTX-Video
library_name: candle
tags:
- ltx-video
- text-to-video
- candle
- rust
- gguf
language:
- en
- ru
license: apache-2.0
---

# LTX-Video in Rust (Candle)

This repository provides a high-performance, native Rust implementation of [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) using the [Candle](https://github.com/huggingface/candle) ML framework.

## Demonstration

| Model | Video | Prompt |
| :--- | :---: | :--- |
| **LTX-Video** | ![Waves and Rocks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/video_without_tiling.gif) | **Prompt:** *The waves crash against the jagged rocks of the shoreline, sending spray high into the air. The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.* |
| **LTX-Video** | ![City or Cat](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/video_with_tiling2.gif) | **Prompt:** *High-quality video generation (768x512) using LTX-Video with VAE Tiling for memory efficiency.* |

## Features

- 🦀 **Native Rust**: No Python dependency required for inference.
- 🚀 **Performance**: Optimized for NVIDIA GPUs with **Flash Attention v2** and **cuDNN**.
- 💾 **Memory Efficient**: Supports **GGUF quantization** for T5-XXL text encoder and **VAE tiling/slicing** for generating 720p+ videos on consumer GPUs.
- 🛠 **Flexible**: Easy to use CLI for video generation and library for custom integration.

## Quick Start

### Installation

Ensure you have Rust and the CUDA Toolkit installed, then:

```bash
git clone https://github.com/FerrisMind/candle-video
cd candle-video
cargo build --release --features flash-attn,cudnn
```

### Video Generation

```bash
cargo run --example ltx-video --release -- \
    --local-weights ./models/ltx-video \
    --prompt "A serene mountain lake at sunset, photorealistic, 4k" \
    --width 768 --height 512 --num-frames 97 \
    --steps 30
```

## Performance & Memory

| Resolution | Frames | VRAM (BF16) | VRAM (VAE Tiling) |
|------------|--------|-------------|-------------------|
| 512x768    | 97     | ~24 GB      | ~16 GB            |
| 384x256    | 25     | ~8 GB       | ~6 GB             |

*Note: Using GGUF T5 encoder saves an additional ~8-12GB of VRAM.*

## Credits

- **Original Model**: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- **Framework**: [HuggingFace Candle](https://github.com/huggingface/candle)
- **Inspiration**: [city96/LTX-Video-gguf](https://huggingface.co/city96/LTX-Video-gguf) (for GGUF support patterns)

---
For more details, visit the main [GitHub Repository](https://github.com/FerrisMind/candle-video).
