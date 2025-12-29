#!/usr/bin/env python3
"""
Debug: Print scheduler timesteps from diffusers for comparison with Rust
"""
import torch
import numpy as np

def show_diffusers_timesteps():
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    # Create scheduler with default LTX config
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_shift=0.95,
        max_shift=2.05,
        num_train_timesteps=1000,
        shift=1.0,
    )
    
    num_inference_steps = 20
    
    # Compute mu (shift) based on sequence length
    # For 512x320, 25 frames: latent dims = 4 x 10 x 16 = 640
    video_sequence_length = 4 * 10 * 16  # = 640
    
    base_seq_len = 256
    max_seq_len = 4096
    base_shift = 0.5
    max_shift_val = 1.15
    
    m = (max_shift_val - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = video_sequence_length * m + b
    
    print(f"Computed mu (shift): {mu:.6f}")
    
    # Set timesteps
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, sigmas=sigmas, mu=mu, device="cpu")
    
    print(f"\nDiffusers sigmas ({len(scheduler.sigmas)}):")
    for i, s in enumerate(scheduler.sigmas):
        print(f"  [{i}] sigma={s:.6f}")
    
    print(f"\nDiffusers timesteps ({len(scheduler.timesteps)}):")
    for i, t in enumerate(scheduler.timesteps):
        print(f"  [{i}] t={t:.6f}")


if __name__ == "__main__":
    show_diffusers_timesteps()
