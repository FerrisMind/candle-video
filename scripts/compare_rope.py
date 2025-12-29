#!/usr/bin/env python3
"""
Compare Rust and diffusers RoPE output.
"""
import numpy as np
import struct
from pathlib import Path


def load_rust_binary(path: str) -> np.ndarray:
    """Load f32 values from Rust binary file."""
    data = Path(path).read_bytes()
    values = struct.unpack(f'{len(data)//4}f', data)
    return np.array(values)


def main():
    output_dir = Path("output")
    
    # Load diffusers output
    diffusers_cos = np.load(output_dir / "diffusers_cos_freqs.npy")
    diffusers_sin = np.load(output_dir / "diffusers_sin_freqs.npy")
    
    print("Diffusers RoPE:")
    print(f"  cos shape: {diffusers_cos.shape}, range: [{diffusers_cos.min():.4f}, {diffusers_cos.max():.4f}]")
    print(f"  sin shape: {diffusers_sin.shape}, range: [{diffusers_sin.min():.4f}, {diffusers_sin.max():.4f}]")
    print(f"  total elements: {diffusers_cos.size}")
    
    # Load Rust output
    rust_cos = load_rust_binary(output_dir / "rust_cos_freqs.bin")
    rust_sin = load_rust_binary(output_dir / "rust_sin_freqs.bin")
    
    print("\nRust RoPE:")
    print(f"  cos size: {rust_cos.size}, range: [{rust_cos.min():.4f}, {rust_cos.max():.4f}]")
    print(f"  sin size: {rust_sin.size}, range: [{rust_sin.min():.4f}, {rust_sin.max():.4f}]")
    
    # Check if sizes match
    if rust_cos.size != diffusers_cos.size:
        print(f"\n⚠ SIZE MISMATCH!")
        print(f"  Diffusers: {diffusers_cos.size}")
        print(f"  Rust: {rust_cos.size}")
        
        # Try to understand the ratio
        if rust_cos.size > 0:
            ratio = diffusers_cos.size / rust_cos.size
            print(f"  Ratio: {ratio:.2f}")
            
            # Still compare first few values
            min_size = min(rust_cos.size, diffusers_cos.flatten().size)
            print(f"\n  First 10 cos values:")
            print(f"    Diffusers: {diffusers_cos.flatten()[:10]}")
            print(f"    Rust:      {rust_cos[:10]}")
            
            print(f"\n  First 10 sin values:")
            print(f"    Diffusers: {diffusers_sin.flatten()[:10]}")
            print(f"    Rust:      {rust_sin[:10]}")
        return
    
    # Compare
    rust_cos = rust_cos.reshape(diffusers_cos.shape)
    rust_sin = rust_sin.reshape(diffusers_sin.shape)
    
    cos_diff = np.abs(diffusers_cos - rust_cos)
    sin_diff = np.abs(diffusers_sin - rust_sin)
    
    print("\nDifference:")
    print(f"  cos max diff: {cos_diff.max():.6f}")
    print(f"  cos mean diff: {cos_diff.mean():.6f}")
    print(f"  sin max diff: {sin_diff.max():.6f}")
    print(f"  sin mean diff: {sin_diff.mean():.6f}")
    
    if cos_diff.max() < 1e-3 and sin_diff.max() < 1e-3:
        print("\n✓ RoPE values match!")
    else:
        print("\n✗ RoPE values differ!")


if __name__ == "__main__":
    main()
