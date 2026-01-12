#!/usr/bin/env python3
"""
Compare Wan VAE decoder outputs between Python (diffusers) and Rust (candle-video).
This script generates reference outputs from Python and compares with Rust outputs.

Usage:
    1. Run this script to generate Python reference outputs
    2. Run the Rust VAE decoder to generate its outputs
    3. Compare the outputs using this script
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)


def get_gpu_memory_gb():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_peak_memory_gb():
    """Get peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def reset_memory():
    """Reset memory counters."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def generate_python_reference(
    vae_path: str,
    latent_frames: int = 9,
    latent_h: int = 60,
    latent_w: int = 60,
    output_dir: str = "debug_outputs",
    seed: int = 42
):
    """Generate Python reference outputs for comparison."""
    from diffusers import AutoencoderKLWan
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("Python VAE Reference Generation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAE
    print("\nLoading VAE...")
    reset_memory()
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    
    print(f"VAE loaded: {get_gpu_memory_gb():.3f} GB")
    
    # Generate deterministic latents
    torch.manual_seed(seed)
    latents = torch.randn(
        1, 16, latent_frames, latent_h, latent_w,
        device=device, dtype=dtype
    )
    
    # Save latents for Rust to use
    latents_path = os.path.join(output_dir, "vae_test_latents.safetensors")
    from safetensors.torch import save_file
    save_file({"latents": latents.cpu()}, latents_path)
    print(f"Saved latents to: {latents_path}")
    
    # Decode
    print("\nDecoding...")
    reset_memory()
    
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    
    peak_mem = get_peak_memory_gb()
    current_mem = get_gpu_memory_gb()
    
    print(f"Decoded shape: {decoded.shape}")
    print(f"Peak memory: {peak_mem:.3f} GB")
    print(f"Current memory: {current_mem:.3f} GB")
    
    # Save decoded output
    decoded_path = os.path.join(output_dir, "vae_decoded_python.safetensors")
    save_file({"decoded": decoded.cpu()}, decoded_path)
    print(f"Saved decoded output to: {decoded_path}")
    
    # Also save as numpy for easier comparison
    np_path = os.path.join(output_dir, "vae_decoded_python.npy")
    np.save(np_path, decoded.cpu().float().numpy())
    print(f"Saved numpy output to: {np_path}")
    
    # Print some statistics
    print(f"\nOutput statistics:")
    print(f"  Min: {decoded.min().item():.6f}")
    print(f"  Max: {decoded.max().item():.6f}")
    print(f"  Mean: {decoded.mean().item():.6f}")
    print(f"  Std: {decoded.std().item():.6f}")
    
    return {
        "latents_path": latents_path,
        "decoded_path": decoded_path,
        "peak_memory_gb": peak_mem,
        "shape": list(decoded.shape),
    }


def compare_outputs(python_path: str, rust_path: str, tolerance: float = 1e-3):
    """Compare Python and Rust VAE outputs."""
    print("=" * 70)
    print("Comparing Python and Rust VAE Outputs")
    print("=" * 70)
    
    # Load outputs
    python_output = np.load(python_path)
    rust_output = np.load(rust_path)
    
    print(f"Python shape: {python_output.shape}")
    print(f"Rust shape: {rust_output.shape}")
    
    if python_output.shape != rust_output.shape:
        print("ERROR: Shape mismatch!")
        return False
    
    # Compute differences
    diff = np.abs(python_output - rust_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nDifference statistics:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    # Find worst locations
    worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"  Worst location: {worst_idx}")
    print(f"  Python value at worst: {python_output[worst_idx]:.6f}")
    print(f"  Rust value at worst: {rust_output[worst_idx]:.6f}")
    
    # Check if within tolerance
    if max_diff < tolerance:
        print(f"\n✓ PASS: Max difference {max_diff:.6f} < tolerance {tolerance}")
        return True
    else:
        print(f"\n✗ FAIL: Max difference {max_diff:.6f} >= tolerance {tolerance}")
        
        # Analyze where differences occur
        high_diff_mask = diff > tolerance
        num_high_diff = np.sum(high_diff_mask)
        pct_high_diff = 100 * num_high_diff / diff.size
        print(f"  Elements above tolerance: {num_high_diff} ({pct_high_diff:.2f}%)")
        
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Wan VAE outputs")
    parser.add_argument("--mode", choices=["generate", "compare"], default="generate",
                       help="Mode: generate Python reference or compare outputs")
    parser.add_argument("--vae-path", default="models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors",
                       help="Path to VAE weights")
    parser.add_argument("--output-dir", default="debug_outputs",
                       help="Output directory")
    parser.add_argument("--latent-frames", type=int, default=9,
                       help="Number of latent frames")
    parser.add_argument("--latent-h", type=int, default=60,
                       help="Latent height")
    parser.add_argument("--latent-w", type=int, default=60,
                       help="Latent width")
    parser.add_argument("--python-output", default=None,
                       help="Path to Python output (for compare mode)")
    parser.add_argument("--rust-output", default=None,
                       help="Path to Rust output (for compare mode)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                       help="Tolerance for comparison")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_python_reference(
            vae_path=args.vae_path,
            latent_frames=args.latent_frames,
            latent_h=args.latent_h,
            latent_w=args.latent_w,
            output_dir=args.output_dir,
        )
    elif args.mode == "compare":
        python_path = args.python_output or os.path.join(args.output_dir, "vae_decoded_python.npy")
        rust_path = args.rust_output or os.path.join(args.output_dir, "vae_decoded_rust.npy")
        compare_outputs(python_path, rust_path, args.tolerance)


if __name__ == "__main__":
    main()
