#!/usr/bin/env python3
"""
Test Wan VAE decoding to verify it works with official weights.
"""

import os
import sys
import torch

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from safetensors.torch import load_file


def is_official_vae_format(state_dict):
    """Check if VAE weights are in official Wan format."""
    for key in state_dict.keys():
        if key.startswith("encoder.conv1") or key.startswith("decoder.conv1"):
            return True
        if ".downsamples." in key or ".upsamples." in key:
            return True
    return False


def main():
    print("=" * 60)
    print("Wan VAE Test")
    print("=" * 60)
    
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load VAE
    print(f"\nLoading VAE from {vae_path}...")
    
    from diffusers import AutoencoderKLWan
    
    # Check format
    state_dict = load_file(vae_path)
    print(f"  Keys sample: {list(state_dict.keys())[:5]}")
    print(f"  Is official format: {is_official_vae_format(state_dict)}")
    
    # Load using diffusers (it handles conversion)
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    
    print(f"  VAE loaded successfully")
    
    # Test decode with small latents
    print("\nTesting VAE decode...")
    
    # Create test latents: [B, C, F, H, W] = [1, 16, 5, 32, 32]
    # This corresponds to 17 frames at 256x256
    latents = torch.randn(1, 16, 5, 32, 32, dtype=dtype, device=device)
    print(f"  Input latents: {latents.shape}")
    
    with torch.no_grad():
        # Decode
        decoded = vae.decode(latents, return_dict=False)[0]
        print(f"  Decoded video: {decoded.shape}")
        print(f"  Value range: [{decoded.min():.3f}, {decoded.max():.3f}]")
    
    print("\n✓ VAE decode successful!")


if __name__ == "__main__":
    main()
