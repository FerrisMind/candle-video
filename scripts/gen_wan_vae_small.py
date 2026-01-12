#!/usr/bin/env python3
"""
Generate small reference data for Wan VAE decoder parity testing.
Uses 256x256 which works in Rust without OOM.
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from safetensors.torch import save_file


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def main():
    print("=" * 70)
    print("Generating Wan VAE Small Reference Data (256x256)")
    print("=" * 70)
    
    vae_path = "models/Wan2.1-T2V-1.3B/vae/wan_2.1_vae.safetensors"
    if not os.path.exists(vae_path):
        print(f"VAE not found at {vae_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load VAE
    print("\nLoading VAE...")
    
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    print(f"VAE loaded. Memory: {get_gpu_memory():.3f} GB")
    
    # Small test case: 256x256 with 17 frames (latent: 5x32x32)
    num_frames = 17
    height = 256
    width = 256
    latent_frames = (num_frames - 1) // 4 + 1  # 5
    latent_h = height // 8  # 32
    latent_w = width // 8  # 32
    
    print(f"\nTest case: {num_frames} frames × {height}×{width}")
    print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    # Create deterministic input
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    
    results = {}
    results["input_latents"] = latents.cpu().float().contiguous()
    
    # Hook to capture intermediate outputs
    captured = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu().float().contiguous()
            elif isinstance(output, tuple):
                captured[name] = output[0].detach().cpu().float().contiguous()
        return hook
    
    # Register hooks on decoder components
    hooks = []
    hooks.append(vae.decoder.conv_in.register_forward_hook(make_hook("decoder_conv_in")))
    hooks.append(vae.decoder.mid_block.register_forward_hook(make_hook("decoder_mid_block")))
    
    for i, up_block in enumerate(vae.decoder.up_blocks):
        hooks.append(up_block.register_forward_hook(make_hook(f"decoder_up_block_{i}")))
    
    hooks.append(vae.decoder.norm_out.register_forward_hook(make_hook("decoder_norm_out")))
    hooks.append(vae.decoder.conv_out.register_forward_hook(make_hook("decoder_conv_out")))
    
    with torch.no_grad():
        # post_quant_conv
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        results["after_post_quant_conv"] = x.cpu().float().contiguous()
        print(f"After post_quant_conv: {x.shape}, mem: {get_gpu_memory():.3f} GB")
        
        # Decode first frame
        print("\nDecoding first frame...")
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        
        vae._conv_idx = [0]
        first_frame = x[:, :, 0:1, :, :]
        results["first_frame_input"] = first_frame.cpu().float().contiguous()
        
        out = vae.decoder(
            first_frame, 
            feat_cache=vae._feat_map, 
            feat_idx=vae._conv_idx, 
            first_chunk=True
        )
        results["first_frame_output"] = out.cpu().float().contiguous()
        print(f"First frame output: {out.shape}, mem: {get_gpu_memory():.3f} GB")
        
        # Save captured for first frame
        for name, tensor in captured.items():
            results[f"frame0_{name}"] = tensor
            print(f"  Captured {name}: {list(tensor.shape)}")
        
        captured.clear()
        
        # Full decode
        print("\nFull decode...")
        vae.clear_cache()
        full_decode = vae.decode(latents, return_dict=False)[0]
        results["full_decode"] = full_decode.cpu().float().contiguous()
        print(f"Full decode: {full_decode.shape}, mem: {get_gpu_memory():.3f} GB")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Save results
    output_path = "gen_wan_vae_small.safetensors"
    save_file(results, output_path)
    print(f"\nSaved reference data to {output_path}")
    
    # Print tensor info
    print("\nSaved tensors:")
    for name, tensor in sorted(results.items()):
        print(f"  {name}: {list(tensor.shape)}")


if __name__ == "__main__":
    main()
