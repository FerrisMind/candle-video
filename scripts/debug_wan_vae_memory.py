#!/usr/bin/env python3
"""
Detailed Wan VAE memory analysis - compare with Rust implementation.
This script shows exactly what memory should be used at each step.
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_gpu_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def tensor_size_gb(t):
    """Calculate tensor size in GB"""
    return t.numel() * t.element_size() / 1024**3


def main():
    print("=" * 70)
    print("Wan VAE Detailed Memory Analysis")
    print("=" * 70)
    
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GB")
    print(f"Dtype: {dtype}")
    
    # Load VAE
    print(f"\n{'='*70}")
    print("Loading VAE...")
    reset_memory()
    
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    
    vae_mem = get_gpu_memory()
    print(f"VAE model size: {vae_mem:.3f} GB")
    
    # Test case: 33 frames at 480x480
    num_frames = 33
    height = 480
    width = 480
    latent_frames = (num_frames - 1) // 4 + 1  # 9
    latent_h = height // 8  # 60
    latent_w = width // 8  # 60
    
    print(f"\n{'='*70}")
    print(f"Test: {num_frames} frames × {height}×{width}")
    print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    # Calculate expected tensor sizes
    latent_size = 1 * 16 * latent_frames * latent_h * latent_w * 2 / 1024**3  # BF16 = 2 bytes
    output_size = 1 * 3 * num_frames * height * width * 2 / 1024**3
    print(f"\nExpected sizes:")
    print(f"  Latent tensor: {latent_size:.3f} GB")
    print(f"  Output tensor: {output_size:.3f} GB")
    
    # Create test latents
    reset_memory()
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    print(f"\nAfter creating latents: {get_gpu_memory():.3f} GB")
    
    # Analyze decoder architecture
    print(f"\n{'='*70}")
    print("Decoder architecture analysis:")
    
    # Count parameters
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    decoder_size = decoder_params * 2 / 1024**3  # BF16
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Decoder size: {decoder_size:.3f} GB")
    
    # Analyze intermediate tensor sizes at each stage
    print(f"\n{'='*70}")
    print("Expected intermediate tensor sizes (per frame):")
    
    # Decoder stages (reversed from encoder)
    # Input: [1, 16, 1, 60, 60]
    # After conv_in: [1, 512, 1, 60, 60]
    # After mid_block: [1, 512, 1, 60, 60]
    # After up_block 0: [1, 256, 1, 120, 120] (2x spatial upsample)
    # After up_block 1: [1, 128, 1, 240, 240] (2x spatial upsample)
    # After up_block 2: [1, 64, 1, 480, 480] (2x spatial upsample)
    # After conv_out: [1, 3, 1, 480, 480]
    
    stages = [
        ("Input", (1, 16, 1, 60, 60)),
        ("After conv_in", (1, 512, 1, 60, 60)),
        ("After mid_block", (1, 512, 1, 60, 60)),
        ("After up_block 0", (1, 256, 1, 120, 120)),
        ("After up_block 1", (1, 128, 1, 240, 240)),
        ("After up_block 2", (1, 64, 1, 480, 480)),
        ("After conv_out", (1, 3, 1, 480, 480)),
    ]
    
    for name, shape in stages:
        size = 1
        for s in shape:
            size *= s
        size_gb = size * 2 / 1024**3  # BF16
        print(f"  {name}: {shape} = {size_gb:.4f} GB")
    
    # The largest intermediate is after up_block 2: [1, 64, 1, 480, 480]
    # But during convolution, we need input + output + intermediate
    # For 3x3x3 conv with padding, intermediate can be 3x larger
    
    print(f"\n{'='*70}")
    print("Memory breakdown during decode:")
    
    with torch.no_grad():
        # Full decode
        reset_memory()
        decoded = vae.decode(latents, return_dict=False)[0]
        
        print(f"  Peak memory: {get_gpu_peak():.3f} GB")
        print(f"  Final memory: {get_gpu_memory():.3f} GB")
        print(f"  Output shape: {decoded.shape}")
        print(f"  Output size: {tensor_size_gb(decoded):.3f} GB")
    
    # Memory budget analysis
    print(f"\n{'='*70}")
    print("Memory budget for 12 GB GPU:")
    print(f"  VAE model: {vae_mem:.3f} GB")
    print(f"  Latents: {latent_size:.3f} GB")
    print(f"  Output: {output_size:.3f} GB")
    print(f"  Cache: ~1.0 GB")
    print(f"  Peak intermediates: ~5.0 GB")
    print(f"  Total peak: ~{vae_mem + latent_size + output_size + 1.0 + 5.0:.1f} GB")
    print(f"  Available: 12.0 GB")
    print(f"  Margin: ~{12.0 - (vae_mem + latent_size + output_size + 1.0 + 5.0):.1f} GB")
    
    # Test without transformer loaded
    print(f"\n{'='*70}")
    print("Memory test (VAE only, no transformer):")
    reset_memory()
    
    # Reload VAE fresh
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    
    print(f"  Before decode: {get_gpu_memory():.3f} GB")
    
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    
    print(f"  Peak: {get_gpu_peak():.3f} GB")
    print(f"  After decode: {get_gpu_memory():.3f} GB")


if __name__ == "__main__":
    main()
