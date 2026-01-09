# candle-ltx-video: Text-to-Video Generation with DiT

LTX-Video is a powerful text-to-video generation model developed by Lightricks, using a Diffusion Transformer (DiT) architecture for high-quality video synthesis.

## Demonstration

| Model | Video | Prompt |
| :--- | :---: | :--- |
| **LTX-Video-0.9.5** | ![Waves and Rocks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/Waves_and_Rocks.gif) | *The waves crash against the jagged rocks of the shoreline, sending spray high into the air. The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.* |
|  | ![woman_with_blood](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/woman_with_blood.gif) | *A woman with blood on her face and a white tank top looks down and to her right, then back up as she speaks. She has dark hair pulled back, light skin, and her face and chest are covered in blood. The camera angle is a close-up, focused on the woman's face and upper torso. The lighting is dim and blue-toned, creating a somber and intense atmosphere. The scene appears to be from a movie or TV show.* |
| |![river](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/river.gif) |  *A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility.* |
| **LTX-Video-0.9.8-2b-distilled** | ![man_walks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/man_walks.gif) | *A man walks towards a window, looks out, and then turns around. He has short, dark hair, dark skin, and is wearing a brown coat over a red and gray scarf. He walks from left to right towards a window, his gaze fixed on something outside. The camera follows him from behind at a medium distance. The room is brightly lit, with white walls and a large window covered by a white curtain. As he approaches the window, he turns his head slightly to the left, then back to the right. He then turns his entire body to the right, facing the window. The camera remains stationary as he stands in front of the window. The scene is captured in real-life footage.* |
| | ![city](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/city.gif) | *The camera pans across a cityscape of tall buildings with a circular building in the center. The camera moves from left to right, showing the tops of the buildings and the circular building in the center. The buildings are various shades of gray and white, and the circular building has a green roof. The camera angle is high, looking down at the city. The lighting is bright, with the sun shining from the upper left, casting shadows from the buildings. The scene is computer-generated imagery.* |
| | ![mountains](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/mountains.gif) | *The camera pans over a snow-covered mountain range, revealing a vast expanse of snow-capped peaks and valleys.The mountains are covered in a thick layer of snow, with some areas appearing almost white while others have a slightly darker, almost grayish hue. The peaks are jagged and irregular, with some rising sharply into the sky while others are more rounded. The valleys are deep and narrow, with steep slopes that are also covered in snow. The trees in the foreground are mostly bare, with only a few leaves remaining on their branches. The sky is overcast, with thick clouds obscuring the sun. The overall impression is one of peace and tranquility, with the snow-covered mountains standing as a testament to the power and beauty of nature.* |
| | ![woman_with_blood](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/woman_with_blood.gif) | *A woman with blood on her face and a white tank top looks down and to her right, then back up as she speaks. She has dark hair pulled back, light skin, and her face and chest are covered in blood. The camera angle is a close-up, focused on the woman's face and upper torso. The lighting is dim and blue-toned, creating a somber and intense atmosphere. The scene appears to be from a movie or TV show.* |

## Model Architecture

- **Transformer**: Diffusion Transformer (DiT) optimized for video generation.
- **Text Encoder**: T5-XXL (supports GGUF quantization for memory efficiency).
- **VAE**: 3D AutoEncoder for spatial and temporal video compression.
- **Scheduler**: Flow Match Euler Discrete Scheduler.

## Running the Model

### Demo
#### 1. T2V
<img src="https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/Demo1.gif" width=75% alt="Demo1" /><br>

#### 1. T2V with key `--seed`

<img src="https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/Demo2.gif" width=75% alt="Demo2" />

### Basic Usage (Auto-download from HuggingFace)

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --prompt "A serene mountain lake at sunset, photorealistic, 4k"
```

### Using Local Weights

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --ltxv-version 0.9.5 \
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
| `--prompt` | The text prompt for video generation | `"A video of a cute cat..."` |
| `--negative-prompt` | Negative prompt for CFG guidance | `""` (empty) |
| `--width` | Width of the generated video (divisible by 32) | `768` |
| `--height` | Height of the generated video (divisible by 32) | `512` |
| `--num-frames` | Number of frames (should be 8n + 1) | `97` |
| `--steps` | Number of denoising steps | From version config (40 for 0.9.5, 8 for distilled) |
| `--guidance-scale` | Classifier-free guidance scale | From version config (3.0 for 0.9.5, 1.0 for distilled) |
| `--ltxv-version` | LTX-Video version (0.9.5, 0.9.8-2b-distilled, etc.) | `0.9.5` |
| `--local-weights` | Path to local model weight directory | Auto-download from HuggingFace |
| `--output-dir` | Directory to save results | `"output"` |
| `--vae-tiling` | Enable spatial VAE tiling | `false` |
| `--vae-slicing` | Enable batch VAE slicing | `false` |
| `--frames` | Save output as individual PNG frames (disables GIF) | `false` |
| `--seed` | Random seed for reproducibility | Random |
| `--cpu` | Run on CPU instead of GPU | `false` |
| `--model-id` | HF model ID (used to download tokenizer if missing) | `"Lightricks/LTX-Video"` |
| `--use-bf16-t5` | Use BF16 T5 instead of GGUF quantized | `false` |
| `--unified-weights` | Path to unified safetensors file (official LTX format) | (None) |
| `--rescaling-scale` | Override rescaling scale from preset | From version config |
| `--stochastic-sampling` | Override stochastic sampling from preset | From version config |

## Video Size Requirements

- **Spatial**: Dimensions must be divisible by 32 (due to 32x VAE compression).
- **Temporal**: Number of frames should ideally follow `8n + 1` (e.g., 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97).

## Performance Notes

- **Flash Attention**: Highly recommended for NVIDIA GPUs to reduce memory usage and increase speed.
- **BF16 Inference**: The model runs in Brain Float 16 by default for optimal performance.
- **Memory**: 97 frames at 512x768 requires ~8-12GB VRAM without optimizations, ~8GB with VAE tiling.

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
