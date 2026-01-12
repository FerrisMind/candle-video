#!/usr/bin/env python3
"""
Debug Wan VAE memory usage step by step.
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


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    print("=" * 70)
    print("Wan VAE Memory Debug (Python)")
    print("=" * 70)
    
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Initial GPU memory: {get_gpu_memory():.3f} GB")
    
    # Load VAE
    print("\nLoading VAE...")
    reset_memory()
    
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    print(f"VAE loaded. GPU memory: {get_gpu_memory():.3f} GB")
    
    # Test 256x256
    print("\n--- Test 256x256 x 17 frames ---")
    num_frames = 17
    height = 256
    width = 256
    latent_frames = (num_frames - 1) // 4 + 1  # 5
    latent_h = height // 8  # 32
    latent_w = width // 8  # 32
    
    print(f"Creating latents [1, 16, {latent_frames}, {latent_h}, {latent_w}]...")
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    print(f"After creating latents: {get_gpu_memory():.3f} GB")
    
    with torch.no_grad():
        print("\nStep-by-step decode:")
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        print(f"After post_quant_conv: {get_gpu_memory():.3f} GB")
        
        out = None
        for i in range(latent_frames):
            vae._conv_idx = [0]
            if i == 0:
                out = vae.decoder(
                    x[:, :, i:i+1, :, :], 
                    feat_cache=vae._feat_map, 
                    feat_idx=vae._conv_idx, 
                    first_chunk=True
                )
            else:
                out_ = vae.decoder(
                    x[:, :, i:i+1, :, :], 
                    feat_cache=vae._feat_map, 
                    feat_idx=vae._conv_idx
                )
                out = torch.cat([out, out_], 2)
            
            # Calculate cache size
            cache_size = sum(
                t.numel() * t.element_size() / 1024**3 
                for t in vae._feat_map if t is not None and isinstance(t, torch.Tensor)
            )
            
            print(f"  Frame {i}: out={out.shape}, mem={get_gpu_memory():.3f} GB, cache={cache_size:.3f} GB")
        
        print(f"\nFinal output: {out.shape}")
        print(f"Final memory: {get_gpu_memory():.3f} GB")
        
        # Clear cache
        vae.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"After clearing cache: {get_gpu_memory():.3f} GB")
    
    # Test 480x480
    print("\n--- Test 480x480 x 33 frames ---")
    num_frames = 33
    height = 480
    width = 480
    latent_frames = (num_frames - 1) // 4 + 1  # 9
    latent_h = height // 8  # 60
    latent_w = width // 8  # 60
    
    # Clear previous
    del out, x, latents
    gc.collect()
    torch.cuda.empty_cache()
    print(f"After clearing previous: {get_gpu_memory():.3f} GB")
    
    print(f"Creating latents [1, 16, {latent_frames}, {latent_h}, {latent_w}]...")
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    print(f"After creating latents: {get_gpu_memory():.3f} GB")
    
    with torch.no_grad():
        print("\nStep-by-step decode:")
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        print(f"After post_quant_conv: {get_gpu_memory():.3f} GB")
        
        out = None
        for i in range(latent_frames):
            vae._conv_idx = [0]
            if i == 0:
                out = vae.decoder(
                    x[:, :, i:i+1, :, :], 
                    feat_cache=vae._feat_map, 
                    feat_idx=vae._conv_idx, 
                    first_chunk=True
                )
            else:
                out_ = vae.decoder(
                    x[:, :, i:i+1, :, :], 
                    feat_cache=vae._feat_map, 
                    feat_idx=vae._conv_idx
                )
                out = torch.cat([out, out_], 2)
            
            # Calculate cache size
            cache_size = sum(
                t.numel() * t.element_size() / 1024**3 
                for t in vae._feat_map if t is not None and isinstance(t, torch.Tensor)
            )
            
            print(f"  Frame {i}: out={out.shape}, mem={get_gpu_memory():.3f} GB, cache={cache_size:.3f} GB")
        
        print(f"\nFinal output: {out.shape}")
        print(f"Final memory: {get_gpu_memory():.3f} GB")


if __name__ == "__main__":
    main()
