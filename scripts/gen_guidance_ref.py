#!/usr/bin/env python3
"""
Test CFG/STG guidance math for LTX-Video.
Generates reference tensors for verifying the guidance formula.
"""

import torch
from safetensors.torch import save_file

def gen_guidance_ref():
    print("Generating CFG/STG guidance reference...")
    
    torch.manual_seed(42)
    
    # Create test noise predictions (batch=1, seq=256, dim=128)
    shape = (1, 256, 128)
    
    noise_pred_uncond = torch.randn(shape)
    noise_pred_text = torch.randn(shape)
    noise_pred_perturb = torch.randn(shape)
    
    guidance_scale = 3.0
    stg_scale = 1.0
    guidance_rescale = 0.0  # Disable for now
    
    print(f"Guidance scale: {guidance_scale}")
    print(f"STG scale: {stg_scale}")
    print(f"Guidance rescale: {guidance_rescale}")
    
    # Python formula (from pipeline_stg_ltx.py lines 811-817):
    # noise_pred = uncond + guidance_scale * (text - uncond) + stg_scale * (text - perturb)
    
    # Step 1: CFG
    diff_cfg = noise_pred_text - noise_pred_uncond
    combined_cfg = noise_pred_uncond + guidance_scale * diff_cfg
    
    # Step 2: STG
    diff_stg = noise_pred_text - noise_pred_perturb
    combined_final = combined_cfg + stg_scale * diff_stg
    
    print(f"\nUncond mean: {noise_pred_uncond.mean().item():.6f}")
    print(f"Text mean: {noise_pred_text.mean().item():.6f}")
    print(f"Perturb mean: {noise_pred_perturb.mean().item():.6f}")
    print(f"CFG diff mean: {diff_cfg.mean().item():.6f}")
    print(f"STG diff mean: {diff_stg.mean().item():.6f}")
    print(f"Combined CFG mean: {combined_cfg.mean().item():.6f}")
    print(f"Combined Final mean: {combined_final.mean().item():.6f}")
    
    # Save for verification
    tensors = {
        "noise_pred_uncond": noise_pred_uncond,
        "noise_pred_text": noise_pred_text,
        "noise_pred_perturb": noise_pred_perturb,
        "guidance_scale": torch.tensor([guidance_scale]),
        "stg_scale": torch.tensor([stg_scale]),
        "combined_cfg": combined_cfg,
        "combined_final": combined_final,
    }
    
    save_file(tensors, "gen_guidance_ref.safetensors")
    print("\nDone. Saved to gen_guidance_ref.safetensors")

if __name__ == "__main__":
    gen_guidance_ref()
