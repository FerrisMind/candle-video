#!/usr/bin/env python3
"""
Test latent normalization/denormalization for LTX-Video.
"""

import torch
from safetensors.torch import save_file

def gen_latent_norm_ref():
    print("Generating latent normalization reference...")
    
    torch.manual_seed(42)
    
    # Typical LTX-Video latent shape: [B, C, F, H, W]
    b, c, f, h, w = 1, 128, 13, 16, 24
    
    # Create test latents
    latents = torch.randn(b, c, f, h, w)
    
    # Typical mean and std from VAE (these would come from vae.latents_mean, vae.latents_std)
    # Using realistic values from LTX-Video VAE config
    latents_mean = torch.randn(c) * 0.1  # small values
    latents_std = torch.ones(c) + torch.randn(c) * 0.1  # around 1.0
    latents_std = latents_std.abs()  # ensure positive
    
    scaling_factor = 1.0  # LTX uses 1.0 by default
    
    print(f"Latents shape: {latents.shape}")
    print(f"Mean shape: {latents_mean.shape}")
    print(f"Std shape: {latents_std.shape}")
    print(f"Scaling factor: {scaling_factor}")
    
    # Normalize: (latents - mean) * scale / std
    mean_reshaped = latents_mean.view(1, -1, 1, 1, 1)
    std_reshaped = latents_std.view(1, -1, 1, 1, 1)
    
    normalized = (latents - mean_reshaped) * scaling_factor / std_reshaped
    
    # Denormalize: latents * std / scale + mean
    denormalized = normalized * std_reshaped / scaling_factor + mean_reshaped
    
    # Check round-trip
    roundtrip_diff = (latents - denormalized).abs().max().item()
    print(f"\nRound-trip max difference: {roundtrip_diff}")
    
    print(f"\nLatents mean: {latents.mean().item():.6f}")
    print(f"Normalized mean: {normalized.mean().item():.6f}")
    print(f"Denormalized mean: {denormalized.mean().item():.6f}")
    
    # Save for verification
    tensors = {
        "latents": latents,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "scaling_factor": torch.tensor([scaling_factor]),
        "normalized": normalized,
        "denormalized": denormalized,
    }
    
    save_file(tensors, "gen_latent_norm_ref.safetensors")
    print("\nDone. Saved to gen_latent_norm_ref.safetensors")

if __name__ == "__main__":
    gen_latent_norm_ref()
