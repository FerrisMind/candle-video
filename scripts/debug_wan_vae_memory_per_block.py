#!/usr/bin/env python3
"""
Profile Wan VAE decoder memory usage per block.
Shows exactly where memory is allocated during decode.
"""

import torch
import gc
from diffusers import AutoencoderKLWan

def get_gpu_mem_gb():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def get_tensor_size_mb(t):
    """Get tensor size in MB."""
    return t.element_size() * t.numel() / 1024**2

def main():
    print("=" * 70)
    print("Wan VAE Decoder Memory Profile Per Block")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load VAE
    print("\nLoading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    
    mem_after_load = get_gpu_mem_gb()
    print(f"VAE loaded. Memory: {mem_after_load:.3f} GB")
    
    # Test case: 480x480, 33 frames
    height, width, num_frames = 480, 480, 33
    latent_h = height // 8
    latent_w = width // 8
    latent_f = (num_frames - 1) // 4 + 1  # 9 for 33 frames
    
    print(f"\nTest case: {num_frames} frames × {height}×{width}")
    print(f"Latent shape: [1, 16, {latent_f}, {latent_h}, {latent_w}]")
    
    # Create random latents
    torch.manual_seed(42)
    z = torch.randn(1, 16, latent_f, latent_h, latent_w, device=device, dtype=dtype)
    print(f"Latent tensor size: {get_tensor_size_mb(z):.2f} MB")
    
    mem_after_latents = get_gpu_mem_gb()
    print(f"Memory after creating latents: {mem_after_latents:.3f} GB")
    
    # Clear cache and prepare
    vae.clear_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("Decoding frame by frame with memory tracking...")
    print("=" * 70)
    
    # Post quant conv
    x = vae.post_quant_conv(z)
    mem_after_pqc = get_gpu_mem_gb()
    print(f"\nAfter post_quant_conv: {x.shape}, mem: {mem_after_pqc:.3f} GB")
    
    out = None
    for i in range(latent_f):
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
        
        mem_now = get_gpu_mem_gb()
        out_size = get_tensor_size_mb(out)
        
        # Calculate cache size
        cache_size = 0
        for entry in vae._feat_map:
            if entry is not None and isinstance(entry, torch.Tensor):
                cache_size += get_tensor_size_mb(entry)
        
        print(f"Frame {i}: out={list(out.shape)}, out_size={out_size:.1f}MB, "
              f"cache={cache_size:.1f}MB, total_mem={mem_now:.3f}GB")
    
    # Final output
    out = torch.clamp(out, min=-1.0, max=1.0)
    vae.clear_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_final = get_gpu_mem_gb()
    print(f"\nFinal output: {list(out.shape)}")
    print(f"Final memory after clear_cache: {mem_final:.3f} GB")
    
    # Show memory breakdown
    print("\n" + "=" * 70)
    print("Memory Breakdown")
    print("=" * 70)
    print(f"VAE model:     {mem_after_load:.3f} GB")
    print(f"Peak during decode: ~{max(mem_after_pqc, mem_now):.3f} GB")
    print(f"Final (after GC):   {mem_final:.3f} GB")

if __name__ == "__main__":
    main()
