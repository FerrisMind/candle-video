#!/usr/bin/env python3
"""
Compare weights from upsampler_detailed with conv3d_ref input/output.
"""

import torch
from safetensors.torch import load_file

def compare():
    print("Loading upsampler_detailed...")
    ups = load_file("gen_upsampler_detailed.safetensors")
    
    print("Loading conv3d_ref...")
    conv = load_file("gen_conv3d_ref.safetensors")
    
    # Get input 
    ups_input = ups["input"]
    conv_input = conv["input"]
    
    print(f"Upsampler input: {ups_input.shape}, mean={ups_input.mean().item():.6f}")
    print(f"Conv3d input: {conv_input.shape}, mean={conv_input.mean().item():.6f}")
    
    # Check if inputs are the same
    input_diff = (ups_input - conv_input).abs().max().item()
    print(f"Input diff: {input_diff}")
    
    # Get weights
    ups_w = ups["upsampler.conv.conv.weight"]
    conv_w = conv["conv.weight"]
    
    print(f"\nUpsampler weight: {ups_w.shape}, mean={ups_w.mean().item():.6f}")
    print(f"Conv3d weight: {conv_w.shape}, mean={conv_w.mean().item():.6f}")
    
    # Check if weights are the same
    weight_diff = (ups_w - conv_w).abs().max().item()
    print(f"Weight diff: {weight_diff}")
    
    # Get outputs
    ups_conv_out = ups["conv_out"]
    conv_out = conv["output"]
    
    print(f"\nUpsampler conv_out: {ups_conv_out.shape}, mean={ups_conv_out.mean().item():.6f}")
    print(f"Conv3d output: {conv_out.shape}, mean={conv_out.mean().item():.6f}")
    
    output_diff = (ups_conv_out - conv_out).abs().max().item()
    print(f"Output diff: {output_diff}")

if __name__ == "__main__":
    compare()
