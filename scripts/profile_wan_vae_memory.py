#!/usr/bin/env python3
"""
Profile Wan VAE memory usage in detail to understand the baseline.
This helps identify where Rust implementation diverges.
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from diffusers import AutoencoderKLWan

def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_peak_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def reset_peak():
    """Reset peak memory counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    
    print("=" * 70)
    print("Wan VAE Memory Profile")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    # Load VAE
    print("\nLoading VAE...")
    reset_peak()
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    print(f"VAE loaded: {get_memory_mb():.1f} MB current, {get_peak_memory_mb():.1f} MB peak")
    
    # Test different resolutions
    test_cases = [
        (33, 256, 256, "256x256"),
        (33, 320, 320, "320x320"),
        (33, 480, 480, "480x480"),
    ]
    
    for num_frames, height, width, name in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test: {name} ({num_frames} frames)")
        print("=" * 70)
        
        # Calculate latent dimensions
        latent_frames = (num_frames - 1) // 4 + 1  # temporal compression
        latent_h = height // 8
        latent_w = width // 8
        
        print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
        
        # Create latents
        gc.collect()
        torch.cuda.empty_cache()
        reset_peak()
        
        latents = torch.randn(
            1, 16, latent_frames, latent_h, latent_w,
            device=device, dtype=dtype
        )
        
        print(f"After creating latents: {get_memory_mb():.1f} MB")
        
        # Decode
        try:
            with torch.no_grad():
                reset_peak()
                output = vae.decode(latents).sample
                
            print(f"Output shape: {output.shape}")
            print(f"After decode: {get_memory_mb():.1f} MB current, {get_peak_memory_mb():.1f} MB peak")
            
            # Calculate tensor sizes
            latent_size_mb = latents.numel() * 2 / 1024 / 1024  # bf16 = 2 bytes
            output_size_mb = output.numel() * 2 / 1024 / 1024
            print(f"Latent tensor size: {latent_size_mb:.2f} MB")
            print(f"Output tensor size: {output_size_mb:.2f} MB")
            
            del output
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM Error: {e}")
        
        del latents
        gc.collect()
        torch.cuda.empty_cache()
    
    # Detailed frame-by-frame analysis for 480x480
    print(f"\n{'=' * 70}")
    print("Frame-by-frame decode analysis (480x480)")
    print("=" * 70)
    
    num_frames = 33
    height, width = 480, 480
    latent_frames = 9
    latent_h, latent_w = 60, 60
    
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak()
    
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, device=device, dtype=dtype)
    
    # Manual frame-by-frame decode (simulating Rust approach)
    print("\nSimulating frame-by-frame decode (like Rust):")
    
    # Get post_quant_conv output
    with torch.no_grad():
        x = vae.post_quant_conv(latents)
    print(f"After post_quant_conv: {get_memory_mb():.1f} MB, peak: {get_peak_memory_mb():.1f} MB")
    
    # Initialize cache
    feat_cache = [None] * 100  # Approximate cache size
    
    outputs = []
    for i in range(latent_frames):
        reset_peak()
        frame = x[:, :, i:i+1, :, :]
        
        with torch.no_grad():
            # This is what the decoder does internally
            decoded = vae.decoder(frame, feat_cache=feat_cache, first_chunk=(i == 0))
        
        outputs.append(decoded)
        
        current_mem = get_memory_mb()
        peak_mem = get_peak_memory_mb()
        cache_size = sum(c.numel() * 2 / 1024 / 1024 if c is not None else 0 for c in feat_cache)
        
        print(f"Frame {i+1}/{latent_frames}: current={current_mem:.1f} MB, peak={peak_mem:.1f} MB, cache={cache_size:.1f} MB")
    
    # Concatenate outputs
    final_output = torch.cat(outputs, dim=2)
    print(f"\nFinal output shape: {final_output.shape}")
    print(f"Final memory: {get_memory_mb():.1f} MB")

if __name__ == "__main__":
    main()
