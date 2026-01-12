#!/usr/bin/env python3
"""
Test Wan transformer shape calculations.

Verifies that the patch embedding and RoPE produce consistent shapes.
"""

import os
import sys
import torch

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)


def main():
    print("=" * 60)
    print("Wan Transformer Shape Test")
    print("=" * 60)
    
    # Target resolution
    num_frames = 81
    height = 480
    width = 832
    
    # Calculate latent dimensions
    latent_frames = (num_frames - 1) // 4 + 1  # 21
    latent_h = height // 8  # 60
    latent_w = width // 8   # 104
    
    print(f"\nVideo: {num_frames} frames × {height}×{width}")
    print(f"Latent: {latent_frames} × {latent_h} × {latent_w}")
    
    # Patch dimensions (1, 2, 2)
    p_t, p_h, p_w = 1, 2, 2
    post_f = latent_frames // p_t  # 21
    post_h = latent_h // p_h       # 30
    post_w = latent_w // p_w       # 52
    seq_len = post_f * post_h * post_w  # 32760
    
    print(f"After patch (1,2,2): {post_f} × {post_h} × {post_w} = {seq_len} tokens")
    
    # Check if model exists
    model_path = "models/Wan2.1-T2V-1.3B"
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        print("Running shape calculations only...")
        
        # Manual shape calculation
        print("\n--- Manual Shape Calculation ---")
        
        # Patch embedding: Conv3d with kernel=(1,2,2), stride=(1,2,2), padding=0
        # Input: [B, 16, 21, 60, 104]
        # Output T: (21 - 1) / 1 + 1 = 21
        # Output H: (60 - 2) / 2 + 1 = 30
        # Output W: (104 - 2) / 2 + 1 = 52
        # Output: [B, 1536, 21, 30, 52]
        
        print(f"Patch embedding input: [1, 16, {latent_frames}, {latent_h}, {latent_w}]")
        out_t = (latent_frames - 1) // 1 + 1
        out_h = (latent_h - 2) // 2 + 1
        out_w = (latent_w - 2) // 2 + 1
        print(f"Patch embedding output: [1, 1536, {out_t}, {out_h}, {out_w}]")
        print(f"Flattened: [1, {out_t * out_h * out_w}, 1536]")
        
        # RoPE calculation
        print(f"\nRoPE seq_len: {post_f} × {post_h} × {post_w} = {seq_len}")
        
        return
    
    print(f"\nLoading transformer from {model_path}...")
    
    from diffusers import WanTransformer3DModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    transformer = WanTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.eval()
    transformer = transformer.to(device)
    
    print("Model loaded.")
    
    # Create test input
    print("\n--- Testing with actual model ---")
    
    hidden_states = torch.randn(
        1, 16, latent_frames, latent_h, latent_w,
        dtype=dtype, device=device
    )
    print(f"Input hidden_states: {hidden_states.shape}")
    
    with torch.no_grad():
        # Test patch embedding
        patch_out = transformer.patch_embedding(hidden_states)
        print(f"Patch embedding output: {patch_out.shape}")
        
        patch_flat = patch_out.flatten(2).transpose(1, 2)
        print(f"Flattened: {patch_flat.shape}")
        
        # Test RoPE
        rotary_emb = transformer.rope(hidden_states)
        print(f"RoPE cos: {rotary_emb[0].shape}")
        print(f"RoPE sin: {rotary_emb[1].shape}")
        
        # Verify shapes match
        rope_seq = rotary_emb[0].shape[1]
        patch_seq = patch_flat.shape[1]
        
        if rope_seq == patch_seq:
            print(f"\n✓ Shapes match: {rope_seq} tokens")
        else:
            print(f"\n✗ Shape mismatch: patch={patch_seq}, rope={rope_seq}")
            print(f"  Difference: {patch_seq - rope_seq}")


if __name__ == "__main__":
    main()
