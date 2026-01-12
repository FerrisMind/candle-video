#!/usr/bin/env python3
"""
Debug script to understand memory usage in Python VAE decode.
This helps identify where Rust implementation uses more memory.
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


def main():
    print("=" * 70)
    print("Debug Python VAE Memory Usage (480x480)")
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
    
    # 480x480 with 33 frames
    num_frames = 33
    height = 480
    width = 480
    latent_frames = (num_frames - 1) // 4 + 1  # 9
    latent_h = height // 8  # 60
    latent_w = width // 8  # 60
    
    print(f"\nTest case: {num_frames} frames × {height}×{width}")
    print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    # Create deterministic input
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    
    print(f"\nMemory after creating latents: {get_gpu_memory():.3f} GB")
    
    # Detailed memory tracking during decode
    print("\n" + "=" * 70)
    print("DETAILED MEMORY TRACKING")
    print("=" * 70)
    
    with torch.no_grad():
        reset_memory()
        vae.clear_cache()
        
        # post_quant_conv
        x = vae.post_quant_conv(latents)
        print(f"\nAfter post_quant_conv: {x.shape}")
        print(f"  Current: {get_gpu_memory():.3f} GB, Peak: {get_peak_memory():.3f} GB")
        
        # Frame by frame decode
        print("\nFrame-by-frame decoding:")
        out = None
        
        for i in range(latent_frames):
            reset_memory()  # Reset peak for each frame
            vae._conv_idx = [0]
            
            frame_input = x[:, :, i:i+1, :, :]
            
            if i == 0:
                out = vae.decoder(
                    frame_input, 
                    feat_cache=vae._feat_map, 
                    feat_idx=vae._conv_idx, 
                    first_chunk=True
                )
            else:
                out_ = vae.decoder(
                    frame_input, 
                    feat_cache=vae._feat_map, 
                    feat_idx=vae._conv_idx
                )
                out = torch.cat([out, out_], 2)
            
            # Calculate cache size
            cache_size = 0
            for t in vae._feat_map:
                if t is not None and isinstance(t, torch.Tensor):
                    cache_size += t.numel() * t.element_size() / 1024**3
            
            print(f"  Frame {i+1}/{latent_frames}:")
            print(f"    Output shape: {out.shape}")
            print(f"    Current mem: {get_gpu_memory():.3f} GB")
            print(f"    Peak mem: {get_peak_memory():.3f} GB")
            print(f"    Cache size: {cache_size:.3f} GB")
            print(f"    Output size: {out.numel() * out.element_size() / 1024**3:.3f} GB")
        
        # Final clamp
        out_clamped = torch.clamp(out, -1.0, 1.0)
        print(f"\nFinal output: {out_clamped.shape}")
        print(f"Final memory: {get_gpu_memory():.3f} GB")
        print(f"Final peak: {get_peak_memory():.3f} GB")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"VAE weights: ~0.24 GB")
    print(f"Latents: {latents.numel() * latents.element_size() / 1024**3:.3f} GB")
    print(f"Output: {out_clamped.numel() * out_clamped.element_size() / 1024**3:.3f} GB")
    print(f"Peak memory during decode: {get_peak_memory():.3f} GB")


if __name__ == "__main__":
    main()
