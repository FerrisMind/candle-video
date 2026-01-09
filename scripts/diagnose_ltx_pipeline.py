"""
Diagnostic script to compare LTX-Video Rust port with Python reference.
Run this to capture reference values for debugging.
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
import sys

# Add LTX-Video to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tp" / "LTX-Video"))

from ltx_video.schedulers.rf import RectifiedFlowScheduler
from diffusers import FlowMatchEulerDiscreteScheduler

def compare_schedulers():
    """Compare RectifiedFlowScheduler vs FlowMatchEulerDiscreteScheduler"""
    print("=" * 60)
    print("Scheduler Comparison")
    print("=" * 60)
    
    # LTX-Video's RectifiedFlowScheduler
    ltx_scheduler = RectifiedFlowScheduler(
        num_train_timesteps=1000,
        shifting="SD3",
        target_shift_terminal=0.1,
        sampler="Uniform",
    )
    
    # Diffusers FlowMatchEulerDiscreteScheduler
    diffusers_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=0.5,
        use_dynamic_shifting=False,
    )
    
    num_steps = 7
    samples_shape = torch.Size([1, 128, 13, 16, 24])  # Typical latent shape
    
    # LTX timesteps
    ltx_scheduler.set_timesteps(num_steps, samples_shape=samples_shape)
    ltx_timesteps = ltx_scheduler.timesteps
    
    # Diffusers timesteps
    diffusers_scheduler.set_timesteps(num_steps, device="cpu")
    diffusers_timesteps = diffusers_scheduler.timesteps
    diffusers_sigmas = diffusers_scheduler.sigmas
    
    print(f"\nLTX RectifiedFlow timesteps ({len(ltx_timesteps)}):")
    print(f"  {ltx_timesteps.tolist()}")
    
    print(f"\nDiffusers FlowMatch timesteps ({len(diffusers_timesteps)}):")
    print(f"  {diffusers_timesteps.tolist()}")
    
    print(f"\nDiffusers sigmas ({len(diffusers_sigmas)}):")
    print(f"  {diffusers_sigmas.tolist()}")
    
    # Test step function
    print("\n" + "=" * 60)
    print("Step Function Comparison")
    print("=" * 60)
    
    sample = torch.randn(1, 128, 2, 4, 4)
    model_output = torch.randn(1, 128, 2, 4, 4)
    
    for i in range(min(3, num_steps)):
        # LTX step
        ltx_t = ltx_timesteps[i]
        ltx_result = ltx_scheduler.step(model_output, ltx_t, sample, return_dict=False)[0]
        
        # Diffusers step
        diff_t = diffusers_timesteps[i]
        diff_result = diffusers_scheduler.step(model_output, diff_t, sample, return_dict=False)[0]
        
        ltx_mean = ltx_result.mean().item()
        diff_mean = diff_result.mean().item()
        
        print(f"\nStep {i}:")
        print(f"  LTX t={ltx_t:.4f}, output mean={ltx_mean:.6f}")
        print(f"  Diffusers t={diff_t:.0f}, output mean={diff_mean:.6f}")


def check_latent_normalization():
    """Check how latents are normalized"""
    print("\n" + "=" * 60)
    print("Latent Normalization")
    print("=" * 60)
    
    # Typical values from LTX-Video
    latents_mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.0778, -0.0277, -0.7841, -0.0648, -0.5467,
        -0.3346, -0.4141, -0.4684, -0.1699, -0.0862, -0.6139, -0.6087, -0.3220,
        # ... (128 channels total in real model)
    ])
    latents_std = torch.tensor([
        6.5886, 6.1893, 5.9493, 5.3848, 5.9116, 6.6298, 6.5816, 6.1768,
        6.0876, 6.3346, 5.9874, 5.7699, 5.6004, 5.9374, 5.8943, 5.6520,
        # ...
    ])
    
    # Create sample latents
    latents = torch.randn(1, 128, 2, 4, 4)
    
    # Method 1: Simple normalize (used in some implementations)
    norm1 = (latents - latents_mean.view(1, -1, 1, 1, 1)) / latents_std.view(1, -1, 1, 1, 1)
    
    # Method 2: Scale only (used in others)
    norm2 = latents / latents_std.view(1, -1, 1, 1, 1)
    
    # Method 3: Offset and scale
    norm3 = (latents * latents_std.view(1, -1, 1, 1, 1)) + latents_mean.view(1, -1, 1, 1, 1)
    
    print(f"Original latents: mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"Norm method 1 (subtract mean, divide std): mean={norm1.mean():.4f}, std={norm1.std():.4f}")
    print(f"Norm method 2 (divide std only): mean={norm2.mean():.4f}, std={norm2.std():.4f}")
    print(f"Norm method 3 (multiply std, add mean): mean={norm3.mean():.4f}, std={norm3.std():.4f}")


def capture_reference_step():
    """Capture a full denoising step for comparison"""
    print("\n" + "=" * 60)
    print("Capturing Reference Step")
    print("=" * 60)
    
    output_dir = Path("reference_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create scheduler
    scheduler = RectifiedFlowScheduler(
        num_train_timesteps=1000,
        shifting="SD3",
        target_shift_terminal=0.1,
        sampler="Uniform",
    )
    
    num_steps = 7
    samples_shape = torch.Size([1, 128, 13, 16, 24])
    scheduler.set_timesteps(num_steps, samples_shape=samples_shape)
    
    # Create reproducible inputs
    torch.manual_seed(42)
    sample = torch.randn(1, 128, 2, 4, 4)
    model_output = torch.randn(1, 128, 2, 4, 4)
    
    results = {}
    
    for i, t in enumerate(scheduler.timesteps):
        result = scheduler.step(model_output, t, sample, return_dict=False, stochastic_sampling=False)[0]
        results[f"step_{i}_input"] = sample.clone()
        results[f"step_{i}_output"] = result.clone()
        results[f"step_{i}_timestep"] = t.unsqueeze(0)
        sample = result  # Update for next step
        print(f"Step {i}: t={t:.4f}, output mean={result.mean():.6f}, std={result.std():.6f}")
    
    results["final_output"] = sample
    results["timesteps"] = scheduler.timesteps
    
    save_file(results, output_dir / "rf_scheduler_steps.safetensors")
    print(f"\nSaved to {output_dir / 'rf_scheduler_steps.safetensors'}")


if __name__ == "__main__":
    compare_schedulers()
    check_latent_normalization()
    capture_reference_step()
