#!/usr/bin/env python3
"""
Generate CausalConv3d reference using MODEL weights (not random).
"""

import torch
from diffusers import AutoencoderKLLTXVideo
from safetensors.torch import save_file

def gen_conv3d_model_weights():
    print("Loading VAE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
    model = AutoencoderKLLTXVideo.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    
    # Get the actual conv from upsampler
    upsampler = model.decoder.up_blocks[0].upsamplers[0]
    conv = upsampler.conv  # This is LTXVideoCausalConv3d
    
    print(f"Conv kernel size: {conv.kernel_size}")
    print(f"Conv is_causal: {conv.is_causal}")
    print(f"Conv.conv weight shape: {conv.conv.weight.shape}")
    
    # Create test input (same as before)
    torch.manual_seed(42)
    x = torch.randn(1, 1024, 8, 16, 16).to(device)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = conv(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean().item():.6f}")
    
    # Save - use same format as we expect in Rust
    tensors = {
        "input": x.cpu(),
        "output": y.cpu(),
        "conv.weight": conv.conv.weight.data.cpu(),
        "conv.bias": conv.conv.bias.data.cpu(),
    }
    
    save_file(tensors, "gen_conv3d_model.safetensors")
    print("\nDone. Saved to gen_conv3d_model.safetensors")

if __name__ == "__main__":
    gen_conv3d_model_weights()
