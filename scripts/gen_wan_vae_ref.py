#!/usr/bin/env python3
"""
Generate reference data for Wan VAE parity testing.
Saves intermediate and final outputs from diffusers VAE decode.
"""

import os
import sys
import torch
import numpy as np

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from safetensors.torch import save_file


def main():
    print("=" * 70)
    print("Generating Wan VAE Reference Data")
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
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded.")
    
    # Test case: small size for quick verification
    # 5 latent frames -> 17 output frames
    latent_frames = 5
    latent_h = 32
    latent_w = 32
    
    print(f"\nTest case: latent shape [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    # Create deterministic input
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    
    results = {}
    results["input_latents"] = latents.cpu().float()
    
    with torch.no_grad():
        # Step 1: post_quant_conv
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        results["after_post_quant_conv"] = x.cpu().float()
        print(f"After post_quant_conv: {x.shape}")
        
        # Step 2: Decode frame by frame (matching _decode implementation)
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        
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
            
            # Save intermediate after each frame
            results[f"after_frame_{i}"] = out.cpu().float()
            print(f"After frame {i}: {out.shape}")
        
        # Step 3: Clamp
        out_clamped = torch.clamp(out, -1.0, 1.0)
        results["final_output"] = out_clamped.cpu().float()
        print(f"Final output: {out_clamped.shape}")
        
        # Also do full decode for comparison
        vae.clear_cache()
        full_decode = vae.decode(latents, return_dict=False)[0]
        results["full_decode"] = full_decode.cpu().float()
        print(f"Full decode: {full_decode.shape}")
    
    # Save results
    output_path = "gen_wan_vae_ref.safetensors"
    save_file(results, output_path)
    print(f"\nSaved reference data to {output_path}")
    
    # Print tensor info
    print("\nSaved tensors:")
    for name, tensor in results.items():
        print(f"  {name}: {list(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
