#!/usr/bin/env python3
"""
Generate detailed reference data for Wan VAE decoder parity testing.
Saves outputs after each decoder block to identify where Rust diverges.
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from safetensors.torch import save_file


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    print("=" * 70)
    print("Generating Wan VAE Decoder Reference Data (480x480)")
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
    
    # Test case: 480x480 with 33 frames (latent: 9x60x60)
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
    
    results = {}
    results["input_latents"] = latents.cpu().float()
    
    print(f"\nMemory after creating latents: {get_gpu_memory():.3f} GB")
    
    with torch.no_grad():
        # Step 1: post_quant_conv
        reset_memory()
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        results["after_post_quant_conv"] = x.cpu().float()
        print(f"After post_quant_conv: {x.shape}, mem: {get_gpu_memory():.3f} GB")
        
        # Step 2: Decode frame by frame
        print("\nFrame-by-frame decoding:")
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        
        out = None
        for i in range(latent_frames):
            reset_memory()
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
            
            # Save intermediate
            results[f"after_frame_{i}"] = out.cpu().float()
            
            # Calculate cache size
            cache_size = sum(
                t.numel() * t.element_size() / 1024**3 
                for t in vae._feat_map if t is not None and isinstance(t, torch.Tensor)
            )
            
            print(f"  Frame {i+1}/{latent_frames}: out={out.shape}, mem={get_gpu_memory():.3f} GB, cache={cache_size:.3f} GB")
        
        # Step 3: Clamp
        out_clamped = torch.clamp(out, -1.0, 1.0)
        results["final_output"] = out_clamped.cpu().float()
        print(f"\nFinal output: {out_clamped.shape}")
    
    # Save results
    output_path = "gen_wan_vae_decoder_ref.safetensors"
    save_file(results, output_path)
    print(f"\nSaved reference data to {output_path}")
    
    # Print tensor info
    print("\nSaved tensors:")
    for name, tensor in results.items():
        print(f"  {name}: {list(tensor.shape)}")


if __name__ == "__main__":
    main()
