#!/usr/bin/env python3
"""
Detailed upsampler debug: capture all intermediate steps.
"""

import torch
from diffusers import AutoencoderKLLTXVideo
from safetensors.torch import save_file

def gen_upsampler_detailed():
    print("Initializing VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
    model = AutoencoderKLLTXVideo.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    
    upsampler = model.decoder.up_blocks[0].upsamplers[0]
    
    print(f"Upsampler stride: {upsampler.stride}")
    print(f"Upsampler residual: {upsampler.residual}")
    print(f"Upsampler upscale_factor: {upsampler.upscale_factor}")
    
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 1024, 8, 16, 16).to(device)
    
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    stride = upsampler.stride
    
    # Step 1: Compute residual (if applicable)
    residual = None
    if upsampler.residual:
        residual = hidden_states.reshape(
            batch_size, -1, stride[0], stride[1], stride[2], num_frames, height, width
        )
        print(f"Residual after reshape: {residual.shape}")
        
        residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        print(f"Residual after permute/flatten: {residual.shape}")
        
        repeats = (stride[0] * stride[1] * stride[2]) // upsampler.upscale_factor
        print(f"Residual repeats: {repeats}")
        
        residual = residual.repeat(1, repeats, 1, 1, 1)
        print(f"Residual after repeat: {residual.shape}")
        
        residual = residual[:, :, stride[0] - 1:]
        print(f"Residual after slice: {residual.shape}")
    
    # Step 2: Conv
    with torch.no_grad():
        conv_out = upsampler.conv(hidden_states)
    print(f"Conv out: {conv_out.shape}")
    
    # Step 3: Main path pixel shuffle
    main_path = conv_out.reshape(
        batch_size, -1, stride[0], stride[1], stride[2], num_frames, height, width
    )
    print(f"Main after reshape: {main_path.shape}")
    
    main_path = main_path.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    print(f"Main after permute/flatten: {main_path.shape}")
    
    main_path = main_path[:, :, stride[0] - 1:]
    print(f"Main after slice: {main_path.shape}")
    
    # Step 4: Add residual
    if residual is not None:
        output = main_path + residual
    else:
        output = main_path
    print(f"Final output: {output.shape}")
    
    # Save
    tensors = {
        "input": hidden_states.cpu(),
        "conv_out": conv_out.cpu(),
        "main_path": main_path.cpu(),
        "output": output.cpu(),
    }
    if residual is not None:
        tensors["residual"] = residual.cpu()
    
    # Save weights
    for name, param in upsampler.named_parameters():
        tensors[f"upsampler.{name}"] = param.data.cpu()
    
    save_file(tensors, "gen_upsampler_detailed.safetensors")
    print("\nDone. Saved to gen_upsampler_detailed.safetensors")
    
    # Print stats
    print(f"\nMain path mean: {main_path.mean().item():.6f}")
    print(f"Residual mean: {residual.mean().item():.6f}" if residual is not None else "No residual")
    print(f"Output mean: {output.mean().item():.6f}")

if __name__ == "__main__":
    gen_upsampler_detailed()
