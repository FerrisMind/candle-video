#!/usr/bin/env python3
"""
Print step-by-step values of RoPE computation in diffusers
to compare with Rust implementation.
"""
import torch
import math
import numpy as np


def prepare_video_coords_debug(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    rope_interpolation_scale,
    base_num_frames: int = 20,
    base_height: int = 2048,
    base_width: int = 2048,
    patch_size: int = 1,
    patch_size_t: int = 1,
):
    """Exact recreation of diffusers LTXVideoRotaryPosEmbed._prepare_video_coords"""
    device = "cpu"
    
    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width, dtype=torch.float32, device=device)
    grid_f = torch.arange(num_frames, dtype=torch.float32, device=device)
    
    print(f"grid_f: {grid_f[:5]}...")
    print(f"grid_h: {grid_h[:5]}...")
    print(f"grid_w: {grid_w[:5]}...")
    
    grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # (3, T, H, W)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # (B, 3, T, H, W)
    
    print(f"grid shape after meshgrid: {grid.shape}")
    print(f"grid[0, 0, :, 0, 0] (T coords at H=0, W=0): {grid[0, 0, :, 0, 0]}")
    print(f"grid[0, 1, 0, :, 0] (H coords at T=0, W=0): {grid[0, 1, 0, :, 0][:5]}...")
    print(f"grid[0, 2, 0, 0, :] (W coords at T=0, H=0): {grid[0, 2, 0, 0, :][:5]}...")
    
    if rope_interpolation_scale is not None:
        grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * patch_size_t / base_num_frames
        grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * patch_size / base_height
        grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * patch_size / base_width
    
    print(f"\nAfter scaling:")
    print(f"grid[0, 0, :, 0, 0] (T coords scaled): {grid[0, 0, :, 0, 0]}")
    print(f"grid[0, 1, 0, :3, 0] (H coords scaled): {grid[0, 1, 0, :3, 0]}")
    print(f"grid[0, 2, 0, 0, :3] (W coords scaled): {grid[0, 2, 0, 0, :3]}")
    
    grid = grid.flatten(2, 4).transpose(1, 2)  # (B, T*H*W, 3)
    
    print(f"\nAfter flatten and transpose: {grid.shape}")
    print(f"First 5 positions (t,h,w coords):")
    for i in range(min(5, grid.shape[1])):
        print(f"  pos {i}: {grid[0, i, :]}")
    
    return grid


def compute_freqs_debug(grid, dim=2048, theta=10000.0):
    """Exact recreation of diffusers LTXVideoRotaryPosEmbed.forward frequency computation"""
    
    freq_dim = dim // 6
    print(f"\nFrequency computation:")
    print(f"dim={dim}, freq_dim={freq_dim}, theta={theta}")
    
    start = 1.0
    end = theta
    # freqs = theta^linspace(log_theta(start), log_theta(end), freq_dim)
    freqs = theta ** torch.linspace(
        math.log(start, theta),
        math.log(end, theta),
        freq_dim,
        device=grid.device,
        dtype=torch.float32,
    )
    print(f"freqs (before *pi/2): {freqs[:5]}...")
    
    freqs = freqs * math.pi / 2.0
    print(f"freqs (after *pi/2): {freqs[:5]}...")
    
    # freqs * (grid * 2 - 1)
    # grid: (B, L, 3), freqs: (freq_dim,)
    # grid.unsqueeze(-1): (B, L, 3, 1)
    # grid * 2 - 1: scale to [-1, 1]
    grid_scaled = grid.unsqueeze(-1) * 2 - 1
    print(f"\ngrid_scaled shape: {grid_scaled.shape}")
    print(f"grid_scaled first pos: {grid_scaled[0, 0, :, 0]}")
    
    freqs_full = freqs * grid_scaled  # (B, L, 3, freq_dim)
    print(f"freqs_full shape: {freqs_full.shape}")
    
    freqs_full = freqs_full.transpose(-1, -2).flatten(2)  # (B, L, 3*freq_dim)
    print(f"freqs after transpose&flatten: {freqs_full.shape}")
    print(f"First 10 freqs at pos 0: {freqs_full[0, 0, :10]}")
    
    cos_freqs = freqs_full.cos().repeat_interleave(2, dim=-1)
    sin_freqs = freqs_full.sin().repeat_interleave(2, dim=-1)
    
    print(f"\ncos_freqs shape: {cos_freqs.shape}")
    print(f"First 10 cos values at pos 0: {cos_freqs[0, 0, :10]}")
    print(f"sin_freqs shape: {sin_freqs.shape}")
    print(f"First 10 sin values at pos 0: {sin_freqs[0, 0, :10]}")
    
    # Padding if dim % 6 != 0
    if dim % 6 != 0:
        pad_size = dim % 6
        cos_padding = torch.ones_like(cos_freqs[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freqs[:, :, :pad_size])
        cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
        sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        print(f"\nAfter padding (dim%6={pad_size}):")
        print(f"cos_freqs final shape: {cos_freqs.shape}")
    
    return cos_freqs, sin_freqs


def main():
    # Parameters matching our test run
    batch_size = 1
    num_frames = 4  # latent frames (25 video frames / 8)
    height = 10     # latent height (320 / 32)
    width = 16      # latent width (512 / 32)
    dim = 2048      # hidden_size
    
    # rope_interpolation_scale from diffusers pipeline
    frame_rate = 25.0
    vae_temporal_compression = 8
    vae_spatial_compression = 32
    rope_interpolation_scale = (
        vae_temporal_compression / frame_rate,  # 8/25 = 0.32
        vae_spatial_compression,                 # 32
        vae_spatial_compression,                 # 32
    )
    
    print("=" * 60)
    print("Diffusers RoPE Debug")
    print("=" * 60)
    print(f"num_frames={num_frames}, height={height}, width={width}")
    print(f"rope_interpolation_scale={rope_interpolation_scale}")
    print()
    
    grid = prepare_video_coords_debug(
        batch_size, num_frames, height, width, rope_interpolation_scale
    )
    
    cos_freqs, sin_freqs = compute_freqs_debug(grid, dim=dim)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Final cos_freqs range: [{cos_freqs.min():.4f}, {cos_freqs.max():.4f}]")
    print(f"Final sin_freqs range: [{sin_freqs.min():.4f}, {sin_freqs.max():.4f}]")
    
    # Save for Rust comparison
    np.save("output/diffusers_cos_freqs.npy", cos_freqs.numpy())
    np.save("output/diffusers_sin_freqs.npy", sin_freqs.numpy())
    print(f"\nSaved freqs to output/diffusers_cos_freqs.npy and output/diffusers_sin_freqs.npy")


if __name__ == "__main__":
    main()
