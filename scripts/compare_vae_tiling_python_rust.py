#!/usr/bin/env python3
"""
Compare Python diffusers VAE tiling vs Rust candle-video VAE tiling.
Generates reference inputs and saves them for Rust to process.
Then compares Python tiled output vs Rust tiled output.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file, load_file

# Add diffusers to path
sys.path.insert(0, "tp/diffusers/src")

def main():
    print("=" * 60)
    print("Python vs Rust VAE Tiling Comparison")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = Path("output/vae_tiling_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VAE model
    print("\n[1/4] Loading VAE model (0.9.5) on CUDA with BF16...")
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not Path(model_path).exists():
        model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae"
    
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    # Print tiling config we're using
    print("\n  Tiling Config (Python diffusers):")
    print(f"    tile_sample_min_height:       {vae.tile_sample_min_height}")
    print(f"    tile_sample_min_width:        {vae.tile_sample_min_width}")
    print(f"    tile_sample_stride_height:    {vae.tile_sample_stride_height}")
    print(f"    tile_sample_stride_width:     {vae.tile_sample_stride_width}")
    print(f"    blend_height = min - stride = {vae.tile_sample_min_height - vae.tile_sample_stride_height}")
    print(f"    blend_width  = min - stride = {vae.tile_sample_min_width - vae.tile_sample_stride_width}")
    
    # Create deterministic test input
    print("\n[2/4] Creating test latents...")
    torch.manual_seed(42)
    
    batch_size = 1
    latent_channels = 128
    
    # Size that triggers tiling: > 512/32 = 16 latent pixels
    latent_t = 5   # 5 latent frames -> 33 video frames
    latent_h = 24  # 24 latent -> 768 pixels
    latent_w = 32  # 32 latent -> 1024 pixels
    
    latents = torch.randn(batch_size, latent_channels, latent_t, latent_h, latent_w, dtype=dtype, device=device)
    temb = torch.tensor([0.05], dtype=dtype, device=device)
    
    print(f"  Latents shape: {latents.shape}")
    print(f"  Output size: {latent_t * 8 - 7}×{latent_h * 32}×{latent_w * 32}")
    
    # Run Python tiled decode
    print("\n[3/4] Running Python tiled decode...")
    vae.use_tiling = True
    vae.use_framewise_decoding = True
    vae.enable_tiling()
    
    with torch.no_grad():
        output_python = vae._decode(latents, temb, return_dict=True).sample
    
    print(f"  Output shape: {output_python.shape}")
    print(f"  Output range: [{output_python.min().item():.4f}, {output_python.max().item():.4f}]")
    
    # Save inputs for Rust
    print("\n[4/4] Saving data for Rust comparison...")
    
    tensors = {
        "latents": latents.cpu().float(),
        "temb": temb.cpu().float(),
        "output_python_tiled": output_python.cpu().float(),
    }
    
    save_path = output_dir / "python_vs_rust_tiling.safetensors"
    save_file(tensors, save_path)
    print(f"  Saved: {save_path}")
    
    # Also save first frame as image for visual comparison
    first_frame = output_python[0, :, 0, :, :].cpu().float()  # [C, H, W]
    first_frame = (first_frame.clamp(-1, 1) + 1) / 2 * 255  # Normalize to [0, 255]
    first_frame = first_frame.permute(1, 2, 0).numpy().astype(np.uint8)  # [H, W, C]
    
    from PIL import Image
    img = Image.fromarray(first_frame, mode='RGB')
    img_path = output_dir / "python_tiled_frame0.png"
    img.save(img_path)
    print(f"  Saved: {img_path}")
    
    print("\n" + "=" * 60)
    print("Now run Rust verification to compare outputs!")
    print("=" * 60)
    print(f"""
Next steps:
1. Create verify_vae_tiling.rs that loads {save_path}
2. Run Rust VAE tiled decode with same inputs
3. Compare output_python_tiled vs output_rust_tiled
""")

if __name__ == "__main__":
    main()
