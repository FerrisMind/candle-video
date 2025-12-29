#!/usr/bin/env python3
"""
Debug first step of DiT denoising to compare with Rust.
Prints key values at each stage.
"""
import torch
import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("Diffusers Single Step Debug")
    print("=" * 60)
    
    from diffusers import LTXPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    
    # Use same parameters as Rust
    width = 512
    height = 320
    num_frames = 25
    num_inference_steps = 20
    guidance_scale = 3.0
    seed = 42
    
    # Compute latent dims
    vae_t_compression = 8
    vae_s_compression = 32
    lat_t = (num_frames - 1) // vae_t_compression + 1  # 4
    lat_h = height // vae_s_compression  # 10
    lat_w = width // vae_s_compression   # 16
    latent_channels = 128
    
    print(f"Video: {width}x{height}x{num_frames}")
    print(f"Latent dims: {lat_t}x{lat_h}x{lat_w}, channels={latent_channels}")
    
    # Initialize scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,  # LTX uses shift=1.0 typically
        use_dynamic_shifting=False,
    )
    
    # Set timesteps - this is what LTX pipeline does
    # Note: for LTX with 0.9.8-distilled, might need different params
    scheduler.set_timesteps(num_inference_steps, device='cpu')
    
    print(f"\nTimesteps ({len(scheduler.timesteps)}):")
    print(f"  First 5: {scheduler.timesteps[:5]}")
    print(f"  Last 5: {scheduler.timesteps[-5:]}")
    
    print(f"\nSigmas ({len(scheduler.sigmas)}):")
    print(f"  First 5: {scheduler.sigmas[:5]}")
    print(f"  Last 5: {scheduler.sigmas[-5:]}")
    
    # Create initial noise
    torch.manual_seed(seed)
    latents = torch.randn(1, latent_channels, lat_t, lat_h, lat_w, dtype=torch.float32)
    
    print(f"\nInitial latents:")
    print(f"  shape: {latents.shape}")
    print(f"  range: [{latents.min():.4f}, {latents.max():.4f}]")
    
    # Pack latents (LTX does this)
    # For patch_size=1: just reshape from 5D to 3D
    # [B, C, T, H, W] -> [B, T*H*W, C]
    packed = latents.permute(0, 2, 3, 4, 1).reshape(1, lat_t * lat_h * lat_w, latent_channels)
    
    print(f"\nPacked latents:")
    print(f"  shape: {packed.shape}")
    print(f"  first 5 values: {packed[0, 0, :5]}")
    
    # First timestep
    t = scheduler.timesteps[0]
    sigma = scheduler.sigmas[0]
    sigma_next = scheduler.sigmas[1]
    dt = sigma_next - sigma
    
    print(f"\nFirst step:")
    print(f"  timestep: {t}")
    print(f"  sigma: {sigma}")
    print(f"  sigma_next: {sigma_next}")
    print(f"  dt: {dt}")
    
    # Simulate model output (random for testing formula)
    torch.manual_seed(123)
    model_output = torch.randn_like(packed)
    
    # Euler step: prev_sample = sample + dt * model_output
    prev_sample = packed + dt * model_output
    
    print(f"\nEuler step result:")
    print(f"  model_output range: [{model_output.min():.4f}, {model_output.max():.4f}]")
    print(f"  prev_sample range: [{prev_sample.min():.4f}, {prev_sample.max():.4f}]")
    
    # Save values for comparison
    out_dir = Path("output/diffusers_step")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "timesteps.npy", scheduler.timesteps.numpy())
    np.save(out_dir / "sigmas.npy", scheduler.sigmas.numpy())
    np.save(out_dir / "initial_latents.npy", latents.numpy())
    
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
