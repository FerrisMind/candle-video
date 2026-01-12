#!/usr/bin/env python3
"""
Profile Wan transformer memory usage step by step.

This script runs through the transformer forward pass step by step,
printing memory usage at each stage to identify where OOM occurs.

Usage:
    python scripts/profile_wan_memory.py
"""

import os
import sys
import torch
import gc

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from safetensors.torch import load_file

# Key mapping from ORIGINAL official Wan format to diffusers format
# Based on tp/diffusers/scripts/convert_wan_to_diffusers.py
# Only used when weights have self_attn/cross_attn naming (original format)
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # Attention mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
}


def is_already_diffusers_format(state_dict):
    """Check if weights are already in diffusers format (have attn1/attn2 naming)."""
    for key in state_dict.keys():
        if ".attn1." in key or ".attn2." in key:
            return True
    return False


def convert_official_to_diffusers(state_dict):
    """Convert official Wan weights to diffusers format.
    
    Handles two cases:
    1. Weights with model.diffusion_model. prefix but already in diffusers naming
       -> Just strip the prefix
    2. Weights in original official format (self_attn, cross_attn)
       -> Full key conversion
    """
    new_state_dict = {}
    prefix = "model.diffusion_model."
    
    # Check if already in diffusers format
    already_diffusers = is_already_diffusers_format(state_dict)
    
    for key, value in state_dict.items():
        new_key = key
        
        # Strip prefix if present
        if new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
        
        # Only apply full conversion if NOT already in diffusers format
        if not already_diffusers:
            for old_pattern, new_pattern in TRANSFORMER_KEYS_RENAME_DICT.items():
                new_key = new_key.replace(old_pattern, new_pattern)
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def is_official_format(model_path):
    """Check if the model is in official Wan format."""
    official_files = [
        "wan2.1_t2v_1.3B_fp16.safetensors",
        "wan2.1_t2v_14B_fp16.safetensors",
    ]
    for f in official_files:
        if os.path.exists(os.path.join(model_path, f)):
            return True
    return False


def load_transformer(model_path, dtype=torch.float16, device="cuda"):
    """Load transformer from either official or diffusers format."""
    from diffusers import WanTransformer3DModel
    
    if is_official_format(model_path):
        print("  Detected: Official Wan format")
        
        weight_files = [
            "wan2.1_t2v_1.3B_fp16.safetensors",
            "wan2.1_t2v_14B_fp16.safetensors",
        ]
        
        weights_path = None
        for f in weight_files:
            path = os.path.join(model_path, f)
            if os.path.exists(path):
                weights_path = path
                break
        
        if weights_path is None:
            raise FileNotFoundError(f"No official Wan weights found in {model_path}")
        
        if "1.3B" in weights_path:
            config = {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            }
        else:
            config = {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            }
        
        transformer = WanTransformer3DModel.from_config(config)
        state_dict = load_file(weights_path)
        converted_state_dict = convert_official_to_diffusers(state_dict)
        transformer.load_state_dict(converted_state_dict, strict=True)
        transformer = transformer.to(dtype=dtype, device=device)
    else:
        print("  Detected: Diffusers format")
        transformer = WanTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        transformer = transformer.to(device)
    
    transformer.eval()
    return transformer


def get_mem():
    """Get current GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_peak_mem():
    """Get peak GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def print_mem(label):
    """Print memory usage with label."""
    print(f"  [{label}] Current: {get_mem():.2f} GB, Peak: {get_peak_mem():.2f} GB")


def main():
    print("=" * 60)
    print("Wan Transformer Memory Profiler")
    print("=" * 60)
    
    model_path = "models/Wan2.1-T2V-1.3B"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    # Load model
    print(f"\nLoading transformer from {model_path}...")
    transformer = load_transformer(model_path, dtype, device)
    print_mem("After model load")
    
    # Test with target resolution: 81 frames, 480x832
    num_frames = 81
    height = 480
    width = 832
    
    latent_frames = (num_frames - 1) // 4 + 1  # 21
    latent_h = height // 8  # 60
    latent_w = width // 8   # 104
    seq_len = latent_frames * (latent_h // 2) * (latent_w // 2)  # 21 * 30 * 52 = 32760
    
    print(f"\nTest configuration:")
    print(f"  Video: {num_frames} frames × {height}×{width}")
    print(f"  Latent: {latent_frames} × {latent_h} × {latent_w}")
    print(f"  Sequence length: {seq_len} tokens")
    
    # Create inputs
    print("\nCreating inputs...")
    torch.cuda.reset_peak_memory_stats()
    
    hidden_states = torch.randn(
        1, 16, latent_frames, latent_h, latent_w,
        dtype=dtype, device=device
    )
    print_mem("After hidden_states")
    
    timestep = torch.tensor([500.0], dtype=dtype, device=device)
    print_mem("After timestep")
    
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=dtype, device=device)
    print_mem("After encoder_hidden_states")
    
    # Step-by-step forward pass
    print("\nStep-by-step forward pass:")
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        # 1. RoPE
        print("\n1. Computing RoPE...")
        rotary_emb = transformer.rope(hidden_states)
        print_mem("After RoPE")
        
        # 2. Patch embedding
        print("\n2. Patch embedding...")
        hs = transformer.patch_embedding(hidden_states)
        hs = hs.flatten(2).transpose(1, 2)
        print_mem("After patch embedding")
        print(f"   Shape: {hs.shape}")
        
        # 3. Condition embedder
        print("\n3. Condition embedder...")
        temb, timestep_proj, enc_text, enc_img = transformer.condition_embedder(
            timestep, encoder_hidden_states, None, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        print_mem("After condition embedder")
        print(f"   temb: {temb.shape}")
        print(f"   timestep_proj: {timestep_proj.shape}")
        print(f"   enc_text: {enc_text.shape}")
        
        # 4. Transformer blocks (one at a time)
        print("\n4. Transformer blocks...")
        for i, block in enumerate(transformer.blocks):
            hs = block(
                hs,
                encoder_hidden_states=enc_text,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
            )
            if i % 5 == 0 or i == len(transformer.blocks) - 1:
                print_mem(f"After block {i}")
        
        # 5. Output norm and projection
        print("\n5. Output norm and projection...")
        
        # Get shift/scale from temb
        shift_scale = transformer.scale_shift_table + temb.unsqueeze(1)
        shift, scale = shift_scale.chunk(2, dim=1)
        shift = shift.squeeze(1).unsqueeze(1)
        scale = scale.squeeze(1).unsqueeze(1)
        
        hs = transformer.norm_out(hs.float()).to(dtype)
        hs = hs * (1 + scale) + shift
        print_mem("After norm_out")
        
        output = transformer.proj_out(hs)
        print_mem("After proj_out")
        print(f"   Output shape: {output.shape}")
        
        # 6. Unpatchify
        print("\n6. Unpatchify...")
        p_t, p_h, p_w = transformer.config.patch_size
        post_f = latent_frames // p_t
        post_h = latent_h // p_h
        post_w = latent_w // p_w
        
        output = output.reshape(1, post_f, post_h, post_w, p_t, p_h, p_w, 16)
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = output.reshape(1, 16, latent_frames, latent_h, latent_w)
        print_mem("After unpatchify")
        print(f"   Final shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Peak memory: {get_peak_mem():.2f} GB")
    print(f"Final memory: {get_mem():.2f} GB")


if __name__ == "__main__":
    main()
