#!/usr/bin/env python3
"""
Generate reference VAE TILING outputs for Rust comparison.
Tests tiled_decode and _temporal_tiled_decode against direct decode.
Saves intermediate tensors for debugging.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

# Add diffusers to path
sys.path.insert(0, "tp/diffusers/src")

def main():
    print("=" * 60)
    print("LTX-Video VAE TILING Verification Generator")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = Path("output/vae_tiling_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VAE model
    print("\n[1/6] Loading VAE model (0.9.5) on CUDA with BF16...")
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not Path(model_path).exists():
        print(f"Path not found: {model_path}")
        model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae"
    
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    # Print tiling config
    print("\n[2/6] VAE Tiling Configuration:")
    print(f"  tile_sample_min_height:       {vae.tile_sample_min_height}")
    print(f"  tile_sample_min_width:        {vae.tile_sample_min_width}")
    print(f"  tile_sample_min_num_frames:   {vae.tile_sample_min_num_frames}")
    print(f"  tile_sample_stride_height:    {vae.tile_sample_stride_height}")
    print(f"  tile_sample_stride_width:     {vae.tile_sample_stride_width}")
    print(f"  tile_sample_stride_num_frames: {vae.tile_sample_stride_num_frames}")
    print(f"  spatial_compression_ratio:    {vae.spatial_compression_ratio}")
    print(f"  temporal_compression_ratio:   {vae.temporal_compression_ratio}")
    
    # Generate deterministic latent input for SPATIAL tiling test
    print("\n[3/6] Testing SPATIAL tiling (tiled_decode)...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    latent_channels = 128
    # Create latents that exceed tile threshold
    # tile_sample_min = 512 / spatial_compression (32) = 16 latent pixels
    # So we need > 16 latent size to trigger tiling
    latent_t = 3  # Small temporal
    latent_h = 20  # > 16 to trigger spatial tiling
    latent_w = 28  # > 16 to trigger spatial tiling
    
    latents_spatial = torch.randn(batch_size, latent_channels, latent_t, latent_h, latent_w, dtype=dtype, device=device)
    temb = torch.tensor([0.05], dtype=dtype, device=device)  # Small timestep
    
    print(f"  Latents shape: {latents_spatial.shape}")
    print(f"  Expected output height: {latent_h * vae.spatial_compression_ratio}")
    print(f"  Expected output width:  {latent_w * vae.spatial_compression_ratio}")
    
    with torch.no_grad():
        # Direct decode (no tiling)
        vae.use_tiling = False
        vae.use_framewise_decoding = False
        output_direct = vae.decoder(latents_spatial, temb)
        print(f"  Direct decode output shape: {output_direct.shape}")
        
        # Tiled decode
        vae.use_tiling = True
        vae.enable_tiling()
        output_tiled = vae.tiled_decode(latents_spatial, temb, return_dict=True).sample
        print(f"  Tiled decode output shape:  {output_tiled.shape}")
        
        # Compare
        diff_spatial = (output_direct - output_tiled).abs()
        print(f"\n  SPATIAL TILING COMPARISON:")
        print(f"    Max absolute difference: {diff_spatial.max().item():.6f}")
        print(f"    Mean absolute difference: {diff_spatial.mean().item():.6f}")
        if diff_spatial.max().item() < 0.01:
            print("    ✅ MATCH - Spatial tiling output matches direct decode")
        else:
            print("    ❌ MISMATCH - Significant difference detected!")
    
    # Test TEMPORAL tiling
    print("\n[4/6] Testing TEMPORAL tiling (_temporal_tiled_decode)...")
    torch.manual_seed(123)
    
    # Create latents that exceed temporal tile threshold
    # tile_sample_min_num_frames / temporal_compression = 16 / 8 = 2 latent frames
    latent_t_temporal = 5  # > 2 to trigger temporal tiling
    latent_h_temporal = 8  # Small spatial (won't trigger spatial tiling)
    latent_w_temporal = 8
    
    latents_temporal = torch.randn(batch_size, latent_channels, latent_t_temporal, latent_h_temporal, latent_w_temporal, dtype=dtype, device=device)
    
    print(f"  Latents shape: {latents_temporal.shape}")
    print(f"  Expected output frames: {(latent_t_temporal - 1) * vae.temporal_compression_ratio + 1}")
    
    with torch.no_grad():
        # Direct decode (no tiling)
        vae.use_tiling = False
        vae.use_framewise_decoding = False
        output_direct_t = vae.decoder(latents_temporal, temb)
        print(f"  Direct decode output shape: {output_direct_t.shape}")
        
        # Temporal tiled decode
        vae.use_framewise_decoding = True
        vae.use_tiling = False
        output_temporal = vae._temporal_tiled_decode(latents_temporal, temb, return_dict=True).sample
        print(f"  Temporal tiled decode output shape: {output_temporal.shape}")
        
        # Compare
        diff_temporal = (output_direct_t - output_temporal).abs()
        print(f"\n  TEMPORAL TILING COMPARISON:")
        print(f"    Max absolute difference: {diff_temporal.max().item():.6f}")
        print(f"    Mean absolute difference: {diff_temporal.mean().item():.6f}")
        if diff_temporal.max().item() < 0.1:
            print("    ✅ MATCH - Temporal tiling output close to direct decode")
        else:
            print("    ⚠️  Note: Some difference is expected due to blending")
    
    # Test COMBINED (spatial + temporal) tiling
    print("\n[5/6] Testing COMBINED tiling (spatial + temporal)...")
    torch.manual_seed(456)
    
    latent_t_combined = 5  # Trigger temporal
    latent_h_combined = 20  # Trigger spatial
    latent_w_combined = 28
    
    latents_combined = torch.randn(batch_size, latent_channels, latent_t_combined, latent_h_combined, latent_w_combined, dtype=dtype, device=device)
    
    print(f"  Latents shape: {latents_combined.shape}")
    
    with torch.no_grad():
        # Direct decode
        vae.use_tiling = False
        vae.use_framewise_decoding = False
        output_direct_c = vae.decoder(latents_combined, temb)
        print(f"  Direct decode output shape: {output_direct_c.shape}")
        
        # Combined tiled decode
        vae.use_tiling = True
        vae.use_framewise_decoding = True
        vae.enable_tiling()
        output_combined = vae._decode(latents_combined, temb, return_dict=True).sample
        print(f"  Combined tiled decode output shape: {output_combined.shape}")
        
        # Compare
        diff_combined = (output_direct_c - output_combined).abs()
        print(f"\n  COMBINED TILING COMPARISON:")
        print(f"    Max absolute difference: {diff_combined.max().item():.6f}")
        print(f"    Mean absolute difference: {diff_combined.mean().item():.6f}")
    
    # Save reference data for Rust
    print("\n[6/6] Saving reference data...")
    
    # Move tensors to CPU for saving
    tensors = {
        "latents_spatial": latents_spatial.cpu().float(),
        "output_spatial_direct": output_direct.cpu().float(),
        "output_spatial_tiled": output_tiled.cpu().float(),
        "latents_temporal": latents_temporal.cpu().float(),
        "output_temporal_direct": output_direct_t.cpu().float(),
        "output_temporal_tiled": output_temporal.cpu().float(),
        "latents_combined": latents_combined.cpu().float(),
        "output_combined_direct": output_direct_c.cpu().float(),
        "output_combined_tiled": output_combined.cpu().float(),
        "temb": temb.cpu().float(),
    }
    
    save_path = output_dir / "vae_tiling_verification.safetensors"
    save_file(tensors, save_path)
    print(f"  Saved: {save_path}")
    
    # Save config as JSON
    import json
    config = {
        "tile_sample_min_height": vae.tile_sample_min_height,
        "tile_sample_min_width": vae.tile_sample_min_width,
        "tile_sample_min_num_frames": vae.tile_sample_min_num_frames,
        "tile_sample_stride_height": vae.tile_sample_stride_height,
        "tile_sample_stride_width": vae.tile_sample_stride_width,
        "tile_sample_stride_num_frames": vae.tile_sample_stride_num_frames,
        "spatial_compression_ratio": vae.spatial_compression_ratio,
        "temporal_compression_ratio": vae.temporal_compression_ratio,
    }
    config_path = output_dir / "tiling_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")
    
    print("\n" + "=" * 60)
    print("VAE Tiling Verification Complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")

if __name__ == "__main__":
    main()
