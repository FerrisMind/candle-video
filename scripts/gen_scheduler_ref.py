#!/usr/bin/env python3
"""
Generate scheduler reference data for LTX-Video pipeline comparison.
Matches Rust FlowMatchEulerDiscreteSchedulerConfig defaults.
"""

import torch
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file

def calculate_mu(latent_seq_len: int, base_seq_len: int = 1024, max_seq_len: int = 4096, 
                 base_shift: float = 0.95, max_shift: float = 2.05) -> float:
    """Calculate mu for dynamic shifting (matches LTX official)."""
    # Linear interpolation between base_shift and max_shift based on sequence length
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = m * latent_seq_len + b
    return mu

def gen_scheduler_ref():
    print("Initializing FlowMatchEulerDiscreteScheduler (LTX default config)...")
    
    # LTX-Video scheduler config from Rust defaults
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,  # Base shift (overridden by dynamic shifting)
        use_dynamic_shifting=True,  # Rust default
        base_shift=0.95,
        max_shift=2.05, 
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        shift_terminal=0.1,  # Rust default - stretches schedule to terminal
    )
    
    num_inference_steps = 40
    
    # Calculate mu based on typical latent sequence length
    # For 768x512 video with 97 frames: latent is ~12x16x24 = 4608 tokens
    # But for smaller test: let's use 8x16x16 = 2048
    latent_seq_len = 2048  
    mu = calculate_mu(latent_seq_len)
    print(f"Calculated mu for seq_len={latent_seq_len}: {mu}")
    
    # Set timesteps with mu
    scheduler.set_timesteps(num_inference_steps, mu=mu)
    
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Timesteps shape: {scheduler.timesteps.shape}")
    print(f"Sigmas shape: {scheduler.sigmas.shape}")
    
    print("\nTimesteps (first 10):", scheduler.timesteps[:10].tolist())
    print("Timesteps (last 10):", scheduler.timesteps[-10:].tolist())
    
    print("\nSigmas (first 10):", scheduler.sigmas[:10].tolist())
    print("Sigmas (last 10):", scheduler.sigmas[-10:].tolist())
    
    # Test a single step
    latent_shape = (1, 128, 8, 16, 16)
    torch.manual_seed(42)
    sample = torch.randn(latent_shape)
    model_output = torch.randn(latent_shape)
    
    t = scheduler.timesteps[0]
    step_output = scheduler.step(model_output, t, sample, return_dict=False)[0]
    
    print(f"\nStep 0 (t={t.item()}):")
    print(f"  Sample mean: {sample.mean().item():.6f}")
    print(f"  Model output mean: {model_output.mean().item():.6f}")
    print(f"  Step output mean: {step_output.mean().item():.6f}")
    
    # Save reference data
    tensors = {
        "timesteps": scheduler.timesteps.float(),
        "sigmas": scheduler.sigmas.float(),
        "sample": sample,
        "model_output": model_output,
        "step_output": step_output,
        "step_t": torch.tensor([t.item()]),
        "mu": torch.tensor([mu]),
    }
    
    save_file(tensors, "gen_scheduler_ref.safetensors")
    print("\nDone. Saved to gen_scheduler_ref.safetensors")

if __name__ == "__main__":
    gen_scheduler_ref()
