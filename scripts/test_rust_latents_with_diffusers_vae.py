#!/usr/bin/env python3
"""
Test: Decode Rust-generated latents with diffusers VAE.

Run this AFTER running Rust inference to generate latents.bin
This will tell us if the problem is in DiT or VAE.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image


def load_latents_bin(path: str, device: str = "cuda") -> torch.Tensor:
    """Load latents saved from Rust in our binary format."""
    with open(path, 'rb') as f:
        # Read header: ndims
        ndims = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        
        # Read shape
        shape = tuple(int(x) for x in np.frombuffer(f.read(8 * ndims), dtype=np.uint64))
        
        # Read data
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    
    tensor = torch.from_numpy(data.copy()).to(device)
    print(f"Loaded latents from {path}")
    print(f"  Shape: {shape}")
    print(f"  Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    return tensor


def save_latents_bin(latents: torch.Tensor, path: str):
    """Save latents in our binary format."""
    data = latents.float().cpu().numpy()
    shape = data.shape
    
    with open(path, 'wb') as f:
        f.write(np.array([len(shape)], dtype=np.uint64).tobytes())
        f.write(np.array(shape, dtype=np.uint64).tobytes())
        f.write(data.tobytes())
    
    print(f"Saved latents to {path}")


def test_decode_with_diffusers_vae(latents_path: str, output_dir: str):
    """Decode latents using diffusers VAE and save result."""
    from diffusers import AutoencoderKLLTXVideo
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load latents
    latents = load_latents_bin(latents_path)
    latents = latents.to(torch.bfloat16)
    
    # Load diffusers VAE
    print("\nLoading diffusers VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae = vae.to("cuda")
    
    # Decode
    print("Decoding with diffusers VAE...")
    decode_timestep = None
    if vae.config.timestep_conditioning:
        decode_timestep = torch.tensor([0.05], device="cuda", dtype=torch.bfloat16)
    
    with torch.no_grad():
        video = vae.decode(latents, decode_timestep, return_dict=False)[0]
    
    print(f"Decoded video shape: {video.shape}")
    
    # Convert to images
    # video shape: [B, C, T, H, W]
    video = video.float().cpu()
    video = (video.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
    video = (video * 255).to(torch.uint8)
    
    # Save first and last frames
    for i, frame_idx in enumerate([0, video.shape[2] - 1]):
        frame = video[0, :, frame_idx, :, :]  # [C, H, W]
        frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        img = Image.fromarray(frame)
        save_path = output_path / f"diffusers_vae_frame_{frame_idx:03d}.png"
        img.save(save_path)
        print(f"Saved frame {frame_idx} to {save_path}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print(f"Check images in {output_path}")
    print("\nIf images look normal: Problem is in Rust VAE")
    print("If images show grid: Problem is in Rust DiT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode Rust latents with diffusers VAE to isolate the bug"
    )
    parser.add_argument(
        "latents_path",
        type=str,
        nargs="?",
        default="output/latents.bin",
        help="Path to latents.bin from Rust inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/vae_test",
        help="Output directory for decoded frames"
    )
    args = parser.parse_args()
    
    test_decode_with_diffusers_vae(args.latents_path, args.output)
