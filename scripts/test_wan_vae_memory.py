#!/usr/bin/env python3
"""
Test Wan VAE memory usage at different resolutions.
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_resolution(vae, height, width, num_frames, dtype, device):
    """Test VAE decode at given resolution."""
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    
    print(f"\n{'='*60}")
    print(f"Testing {height}x{width} with {num_frames} frames")
    print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    reset_memory()
    
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    
    print(f"After creating latents: {get_gpu_memory():.3f} GB")
    
    try:
        with torch.no_grad():
            vae.clear_cache()
            output = vae.decode(latents, return_dict=False)[0]
            
        print(f"Output shape: {output.shape}")
        print(f"Current memory: {get_gpu_memory():.3f} GB")
        print(f"Peak memory: {get_peak_memory():.3f} GB")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        print(f"Peak memory before failure: {get_peak_memory():.3f} GB")
        return False


def main():
    print("=" * 70)
    print("Wan VAE Memory Test")
    print("=" * 70)
    
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load VAE
    print("\nLoading VAE...")
    reset_memory()
    
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    print(f"VAE loaded. Memory: {get_gpu_memory():.3f} GB")
    
    # Test different resolutions
    test_cases = [
        (256, 256, 17),   # Small
        (320, 320, 17),   # Medium-small
        (384, 384, 17),   # Medium
        (416, 416, 17),   # Medium-large
        (448, 448, 17),   # Large
        (480, 480, 17),   # Target
        (480, 480, 33),   # Target with more frames
    ]
    
    for height, width, num_frames in test_cases:
        success = test_resolution(vae, height, width, num_frames, dtype, device)
        if not success:
            print(f"\nStopping at {height}x{width}")
            break
    
    print("\n" + "=" * 70)
    print("Test complete")


if __name__ == "__main__":
    main()
