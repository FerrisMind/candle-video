#!/usr/bin/env python3
"""
Debug Wan VAE by comparing Python diffusers implementation with our understanding.
This helps identify where memory issues might be in the Rust implementation.
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from safetensors.torch import load_file


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_gpu_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    print("=" * 70)
    print("Wan VAE Debug - Step by Step Memory Analysis")
    print("=" * 70)
    
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    device = torch.device("cuda")
    dtype = torch.bfloat16  # Match Rust BF16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GB")
    
    # Load VAE
    print(f"\n{'='*70}")
    print("Loading VAE...")
    reset_memory()
    
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    
    print(f"VAE loaded. Memory: {get_gpu_memory():.3f} GB")
    
    # Test case: 33 frames at 480x480 (latent: 9x60x60)
    num_frames = 33
    height = 480
    width = 480
    latent_frames = (num_frames - 1) // 4 + 1  # 9
    latent_h = height // 8  # 60
    latent_w = width // 8  # 60
    
    print(f"\n{'='*70}")
    print(f"Test: {num_frames} frames × {height}×{width}")
    print(f"Latent: {latent_frames} × {latent_h} × {latent_w}")
    
    # Create test latents
    reset_memory()
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    print(f"\nAfter creating latents: {get_gpu_memory():.3f} GB")
    
    # Step-by-step decode analysis
    print(f"\n{'='*70}")
    print("Step-by-step decode analysis:")
    
    with torch.no_grad():
        # Step 1: post_quant_conv
        reset_memory()
        x = vae.post_quant_conv(latents)
        print(f"1. post_quant_conv: {get_gpu_memory():.3f} GB, peak: {get_gpu_peak():.3f} GB")
        print(f"   Output shape: {x.shape}")
        
        # Step 2: Clear cache
        vae.clear_cache()
        
        # Step 3: Decode frame by frame
        print(f"\n2. Frame-by-frame decoding:")
        out = None
        for i in range(latent_frames):
            reset_memory()
            vae._conv_idx = [0]
            frame_input = x[:, :, i:i+1, :, :]
            
            if i == 0:
                out = vae.decoder(frame_input, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)
            else:
                out_ = vae.decoder(frame_input, feat_cache=vae._feat_map, feat_idx=vae._conv_idx)
                out = torch.cat([out, out_], 2)
            
            cache_size = sum(
                t.numel() * t.element_size() / 1024**3 
                for t in vae._feat_map if t is not None and isinstance(t, torch.Tensor)
            )
            print(f"   Frame {i+1}/{latent_frames}: mem={get_gpu_memory():.3f} GB, peak={get_gpu_peak():.3f} GB, cache={cache_size:.3f} GB, out={out.shape}")
        
        print(f"\n3. Final output shape: {out.shape}")
        print(f"   Final memory: {get_gpu_memory():.3f} GB")
        
        # Step 4: Clamp
        reset_memory()
        out = torch.clamp(out, -1.0, 1.0)
        print(f"\n4. After clamp: {get_gpu_memory():.3f} GB, peak: {get_gpu_peak():.3f} GB")
    
    # Full decode for comparison
    print(f"\n{'='*70}")
    print("Full decode (for comparison):")
    reset_memory()
    
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    
    print(f"Full decode: mem={get_gpu_memory():.3f} GB, peak={get_gpu_peak():.3f} GB")
    print(f"Output shape: {decoded.shape}")
    
    # Check cache structure
    print(f"\n{'='*70}")
    print("Cache analysis:")
    print(f"Number of cache entries: {len(vae._feat_map)}")
    cache_entries = [(i, type(t).__name__, t.shape if hasattr(t, 'shape') else 'N/A') 
                     for i, t in enumerate(vae._feat_map) if t is not None]
    print(f"Non-None entries: {len(cache_entries)}")
    for i, ttype, shape in cache_entries[:10]:
        print(f"  [{i}] {ttype}: {shape}")
    if len(cache_entries) > 10:
        print(f"  ... and {len(cache_entries) - 10} more")


if __name__ == "__main__":
    main()
