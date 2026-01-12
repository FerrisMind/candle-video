#!/usr/bin/env python3
"""
Generate step-by-step reference data for Wan VAE decoder parity testing.
Saves outputs after each decoder block to identify where Rust diverges.

This script hooks into the decoder to capture intermediate outputs.
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


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    print("=" * 70)
    print("Generating Wan VAE Step-by-Step Reference Data")
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
    reset_memory()
    
    from diffusers import AutoencoderKLWan
    
    vae = AutoencoderKLWan.from_single_file(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    print(f"VAE loaded. Memory: {get_gpu_memory():.3f} GB")
    
    # Test case: 480x480 with 33 frames (latent: 9x60x60)
    # Use smaller size first to verify parity
    num_frames = 33
    height = 480
    width = 480
    latent_frames = (num_frames - 1) // 4 + 1  # 9
    latent_h = height // 8  # 60
    latent_w = width // 8  # 60
    
    print(f"\nTest case: {num_frames} frames × {height}×{width}")
    print(f"Latent shape: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
    
    # Create deterministic input
    torch.manual_seed(42)
    latents = torch.randn(1, 16, latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    
    results = {}
    results["input_latents"] = latents.cpu().float()
    
    print(f"\nMemory after creating latents: {get_gpu_memory():.3f} GB")
    
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
    
    # Hook conv_in
    hooks.append(vae.decoder.conv_in.register_forward_hook(make_hook("decoder_conv_in")))
    
    # Hook mid_block
    hooks.append(vae.decoder.mid_block.register_forward_hook(make_hook("decoder_mid_block")))
    
    # Hook each up_block
    for i, up_block in enumerate(vae.decoder.up_blocks):
        hooks.append(up_block.register_forward_hook(make_hook(f"decoder_up_block_{i}")))
        # Also hook individual resnets
        for j, resnet in enumerate(up_block.resnets):
            hooks.append(resnet.register_forward_hook(make_hook(f"decoder_up_block_{i}_resnet_{j}")))
        # Hook upsampler if exists
        if hasattr(up_block, 'upsamplers') and up_block.upsamplers is not None:
            for k, ups in enumerate(up_block.upsamplers):
                hooks.append(ups.register_forward_hook(make_hook(f"decoder_up_block_{i}_upsampler_{k}")))
    
    # Hook norm_out and conv_out
    hooks.append(vae.decoder.norm_out.register_forward_hook(make_hook("decoder_norm_out")))
    hooks.append(vae.decoder.conv_out.register_forward_hook(make_hook("decoder_conv_out")))
    
    with torch.no_grad():
        # Step 1: post_quant_conv
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        results["after_post_quant_conv"] = x.cpu().float()
        print(f"After post_quant_conv: {x.shape}, mem: {get_gpu_memory():.3f} GB")
        
        # Step 2: Decode frame by frame (single frame to capture intermediate states)
        print("\nDecoding first frame to capture intermediate states...")
        vae.clear_cache()
        x = vae.post_quant_conv(latents)
        
        # Process just first frame
        vae._conv_idx = [0]
        first_frame = x[:, :, 0:1, :, :]
        results["first_frame_input"] = first_frame.cpu().float()
        
        out = vae.decoder(
            first_frame, 
            feat_cache=vae._feat_map, 
            feat_idx=vae._conv_idx, 
            first_chunk=True
        )
        results["first_frame_output"] = out.cpu().float()
        print(f"First frame output: {out.shape}, mem: {get_gpu_memory():.3f} GB")
        
        # Save all captured intermediate outputs
        for name, tensor in captured.items():
            results[f"frame0_{name}"] = tensor
            print(f"  Captured {name}: {list(tensor.shape)}")
        
        # Clear captured for next frame
        captured.clear()
        
        # Process second frame
        print("\nDecoding second frame...")
        vae._conv_idx = [0]
        second_frame = x[:, :, 1:2, :, :]
        results["second_frame_input"] = second_frame.cpu().float()
        
        out2 = vae.decoder(
            second_frame, 
            feat_cache=vae._feat_map, 
            feat_idx=vae._conv_idx, 
            first_chunk=False
        )
        results["second_frame_output"] = out2.cpu().float()
        print(f"Second frame output: {out2.shape}, mem: {get_gpu_memory():.3f} GB")
        
        # Save captured for second frame
        for name, tensor in captured.items():
            results[f"frame1_{name}"] = tensor
            print(f"  Captured {name}: {list(tensor.shape)}")
        
        # Full decode for comparison
        print("\nFull decode...")
        vae.clear_cache()
        full_decode = vae.decode(latents, return_dict=False)[0]
        results["full_decode"] = full_decode.cpu().float()
        print(f"Full decode: {full_decode.shape}, mem: {get_gpu_memory():.3f} GB")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Save results
    output_path = "gen_wan_vae_stepwise.safetensors"
    save_file(results, output_path)
    print(f"\nSaved reference data to {output_path}")
    
    # Print tensor info
    print("\nSaved tensors:")
    for name, tensor in sorted(results.items()):
        print(f"  {name}: {list(tensor.shape)}")


if __name__ == "__main__":
    main()
