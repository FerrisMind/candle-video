#!/usr/bin/env python3
"""
Capture reference tensors for CFG (Classifier-Free Guidance) parity testing.

This script generates reference data for:
1. Basic CFG formula: noise_uncond + guidance_scale * (noise_cond - noise_uncond)
2. rescale_noise_cfg function with std computation over dims 1..N

Requirements: 7.1, 7.2
"""

import torch
import numpy as np
from safetensors.torch import save_file


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality.
    Based on Section 3.4 from https://huggingface.co/papers/2305.08891
    
    This is the exact implementation from diffusers.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def apply_cfg(noise_uncond, noise_cond, guidance_scale):
    """
    Apply Classifier-Free Guidance.
    Formula: noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    """
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


def main():
    print("Capturing CFG parity reference tensors...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    tensors = {}
    
    # Test case 1: Basic CFG formula with small tensors
    # Shape: [B, S, D] typical for packed latents
    batch_size = 1
    seq_len = 64
    hidden_dim = 128
    
    noise_uncond_1 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
    noise_cond_1 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
    guidance_scale_1 = 7.5
    
    cfg_result_1 = apply_cfg(noise_uncond_1, noise_cond_1, guidance_scale_1)
    
    tensors["test1_noise_uncond"] = noise_uncond_1
    tensors["test1_noise_cond"] = noise_cond_1
    tensors["test1_guidance_scale"] = torch.tensor([guidance_scale_1], dtype=torch.float32)
    tensors["test1_cfg_result"] = cfg_result_1
    
    print(f"Test 1: Basic CFG [B={batch_size}, S={seq_len}, D={hidden_dim}], scale={guidance_scale_1}")
    print(f"  noise_uncond range: [{noise_uncond_1.min():.4f}, {noise_uncond_1.max():.4f}]")
    print(f"  noise_cond range: [{noise_cond_1.min():.4f}, {noise_cond_1.max():.4f}]")
    print(f"  cfg_result range: [{cfg_result_1.min():.4f}, {cfg_result_1.max():.4f}]")
    
    # Test case 2: CFG with different guidance scale
    guidance_scale_2 = 3.0
    cfg_result_2 = apply_cfg(noise_uncond_1, noise_cond_1, guidance_scale_2)
    
    tensors["test2_guidance_scale"] = torch.tensor([guidance_scale_2], dtype=torch.float32)
    tensors["test2_cfg_result"] = cfg_result_2
    
    print(f"Test 2: CFG with scale={guidance_scale_2}")
    print(f"  cfg_result range: [{cfg_result_2.min():.4f}, {cfg_result_2.max():.4f}]")
    
    # Test case 3: CFG with guidance_scale = 1.0 (should equal noise_cond)
    guidance_scale_3 = 1.0
    cfg_result_3 = apply_cfg(noise_uncond_1, noise_cond_1, guidance_scale_3)
    
    tensors["test3_guidance_scale"] = torch.tensor([guidance_scale_3], dtype=torch.float32)
    tensors["test3_cfg_result"] = cfg_result_3
    
    # Verify: when scale=1, result should equal noise_cond
    diff_to_cond = (cfg_result_3 - noise_cond_1).abs().max().item()
    print(f"Test 3: CFG with scale=1.0 (should equal noise_cond)")
    print(f"  Max diff to noise_cond: {diff_to_cond:.2e}")
    
    # Test case 4: rescale_noise_cfg function
    # First apply CFG, then rescale
    guidance_scale_4 = 7.5
    guidance_rescale_4 = 0.7
    
    cfg_before_rescale = apply_cfg(noise_uncond_1, noise_cond_1, guidance_scale_4)
    cfg_after_rescale = rescale_noise_cfg(cfg_before_rescale, noise_cond_1, guidance_rescale_4)
    
    tensors["test4_cfg_before_rescale"] = cfg_before_rescale
    tensors["test4_guidance_rescale"] = torch.tensor([guidance_rescale_4], dtype=torch.float32)
    tensors["test4_cfg_after_rescale"] = cfg_after_rescale
    
    print(f"Test 4: rescale_noise_cfg with guidance_rescale={guidance_rescale_4}")
    print(f"  Before rescale range: [{cfg_before_rescale.min():.4f}, {cfg_before_rescale.max():.4f}]")
    print(f"  After rescale range: [{cfg_after_rescale.min():.4f}, {cfg_after_rescale.max():.4f}]")
    
    # Test case 5: rescale_noise_cfg with different rescale values
    guidance_rescale_5 = 0.0  # Should return original cfg
    cfg_after_rescale_5 = rescale_noise_cfg(cfg_before_rescale, noise_cond_1, guidance_rescale_5)
    
    tensors["test5_guidance_rescale"] = torch.tensor([guidance_rescale_5], dtype=torch.float32)
    tensors["test5_cfg_after_rescale"] = cfg_after_rescale_5
    
    diff_to_original = (cfg_after_rescale_5 - cfg_before_rescale).abs().max().item()
    print(f"Test 5: rescale_noise_cfg with guidance_rescale=0.0 (should equal original)")
    print(f"  Max diff to original: {diff_to_original:.2e}")
    
    # Test case 6: Larger tensor with 5D shape [B, C, F, H, W] (unpacked latents)
    batch_size_6 = 1
    channels = 128
    frames = 13
    height = 16
    width = 24
    
    torch.manual_seed(123)
    noise_uncond_6 = torch.randn(batch_size_6, channels, frames, height, width, dtype=torch.float32)
    noise_cond_6 = torch.randn(batch_size_6, channels, frames, height, width, dtype=torch.float32)
    guidance_scale_6 = 5.0
    guidance_rescale_6 = 0.5
    
    cfg_result_6 = apply_cfg(noise_uncond_6, noise_cond_6, guidance_scale_6)
    cfg_rescaled_6 = rescale_noise_cfg(cfg_result_6, noise_cond_6, guidance_rescale_6)
    
    tensors["test6_noise_uncond"] = noise_uncond_6
    tensors["test6_noise_cond"] = noise_cond_6
    tensors["test6_guidance_scale"] = torch.tensor([guidance_scale_6], dtype=torch.float32)
    tensors["test6_guidance_rescale"] = torch.tensor([guidance_rescale_6], dtype=torch.float32)
    tensors["test6_cfg_result"] = cfg_result_6
    tensors["test6_cfg_rescaled"] = cfg_rescaled_6
    
    print(f"Test 6: 5D tensor [B={batch_size_6}, C={channels}, F={frames}, H={height}, W={width}]")
    print(f"  CFG result range: [{cfg_result_6.min():.4f}, {cfg_result_6.max():.4f}]")
    print(f"  Rescaled result range: [{cfg_rescaled_6.min():.4f}, {cfg_rescaled_6.max():.4f}]")
    
    # Test case 7: Verify std computation over dims 1..N
    # This is critical for rescale_noise_cfg parity
    std_text_6 = noise_cond_6.std(dim=list(range(1, noise_cond_6.ndim)), keepdim=True)
    std_cfg_6 = cfg_result_6.std(dim=list(range(1, cfg_result_6.ndim)), keepdim=True)
    
    tensors["test7_std_text"] = std_text_6
    tensors["test7_std_cfg"] = std_cfg_6
    
    print(f"Test 7: std computation verification")
    print(f"  std_text shape: {std_text_6.shape}, value: {std_text_6.item():.6f}")
    print(f"  std_cfg shape: {std_cfg_6.shape}, value: {std_cfg_6.item():.6f}")
    
    # Test case 8: Batch size > 1
    batch_size_8 = 2
    torch.manual_seed(456)
    noise_uncond_8 = torch.randn(batch_size_8, seq_len, hidden_dim, dtype=torch.float32)
    noise_cond_8 = torch.randn(batch_size_8, seq_len, hidden_dim, dtype=torch.float32)
    guidance_scale_8 = 7.5
    guidance_rescale_8 = 0.7
    
    cfg_result_8 = apply_cfg(noise_uncond_8, noise_cond_8, guidance_scale_8)
    cfg_rescaled_8 = rescale_noise_cfg(cfg_result_8, noise_cond_8, guidance_rescale_8)
    
    tensors["test8_noise_uncond"] = noise_uncond_8
    tensors["test8_noise_cond"] = noise_cond_8
    tensors["test8_guidance_scale"] = torch.tensor([guidance_scale_8], dtype=torch.float32)
    tensors["test8_guidance_rescale"] = torch.tensor([guidance_rescale_8], dtype=torch.float32)
    tensors["test8_cfg_result"] = cfg_result_8
    tensors["test8_cfg_rescaled"] = cfg_rescaled_8
    
    # Verify per-batch std
    std_text_8 = noise_cond_8.std(dim=list(range(1, noise_cond_8.ndim)), keepdim=True)
    std_cfg_8 = cfg_result_8.std(dim=list(range(1, cfg_result_8.ndim)), keepdim=True)
    
    tensors["test8_std_text"] = std_text_8
    tensors["test8_std_cfg"] = std_cfg_8
    
    print(f"Test 8: Batch size > 1 [B={batch_size_8}]")
    print(f"  std_text shape: {std_text_8.shape}")
    print(f"  std_text values: {std_text_8.squeeze().tolist()}")
    print(f"  std_cfg values: {std_cfg_8.squeeze().tolist()}")
    
    # Save all tensors
    output_path = "gen_cfg_parity.safetensors"
    save_file(tensors, output_path)
    print(f"\nSaved {len(tensors)} tensors to {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print("CFG Formula: noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)")
    print("Rescale Formula: noise_cfg = guidance_rescale * (noise_cfg * std_text/std_cfg) + (1-guidance_rescale) * noise_cfg")
    print("std computation: std over dims 1..N with keepdim=True")


if __name__ == "__main__":
    main()
