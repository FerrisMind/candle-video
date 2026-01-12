#!/usr/bin/env python3
"""
Debug script to capture Wan transformer forward pass for parity testing.

This script runs a minimal forward pass through the Wan transformer
and saves intermediate tensors for comparison with the Rust implementation.

Supports both:
- Official Wan weights (wan2.1_t2v_1.3B_fp16.safetensors)
- Diffusers format (transformer/diffusion_pytorch_model.safetensors)

Usage:
    python scripts/debug_wan_transformer.py

Output: gen_wan_transformer_debug.safetensors
"""

import os
import sys
import torch
from safetensors.torch import save_file, load_file

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

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
    # Official format has files like wan2.1_t2v_1.3B_fp16.safetensors in root
    # Diffusers format has transformer/diffusion_pytorch_model.safetensors
    official_files = [
        "wan2.1_t2v_1.3B_fp16.safetensors",
        "wan2.1_t2v_14B_fp16.safetensors",
    ]
    for f in official_files:
        if os.path.exists(os.path.join(model_path, f)):
            return True
    return False


def load_transformer_from_official(model_path, dtype=torch.float16, device="cuda"):
    """Load transformer from official Wan weights."""
    from diffusers import WanTransformer3DModel
    
    # Find the weights file
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
    
    print(f"  Loading official weights from {weights_path}")
    
    # Determine config based on file name
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
    
    # Create model with config
    transformer = WanTransformer3DModel.from_config(config)
    
    # Load and convert weights
    state_dict = load_file(weights_path)
    converted_state_dict = convert_official_to_diffusers(state_dict)
    
    # Load weights
    transformer.load_state_dict(converted_state_dict, strict=True)
    transformer = transformer.to(dtype=dtype, device=device)
    transformer.eval()
    
    return transformer


def load_transformer_from_diffusers(model_path, dtype=torch.float16, device="cuda"):
    """Load transformer from diffusers format."""
    from diffusers import WanTransformer3DModel
    
    transformer = WanTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer = transformer.to(device)
    transformer.eval()
    
    return transformer


def main():
    print("=" * 60)
    print("Wan Transformer Debug - Minimal Forward Pass")
    print("=" * 60)
    
    # Check for model
    model_path = "models/Wan2.1-T2V-1.3B"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download the model first.")
        return
    
    # Use smaller dimensions for testing
    # This should fit in ~2GB VRAM
    batch_size = 1
    num_frames = 17  # -> 5 latent frames
    height = 256     # -> 32 latent height -> 16 patches
    width = 256      # -> 32 latent width -> 16 patches
    
    # Calculate latent dimensions
    latent_frames = (num_frames - 1) // 4 + 1  # 5
    latent_h = height // 8  # 32
    latent_w = width // 8   # 32
    
    # After patch (1, 2, 2)
    seq_len = latent_frames * (latent_h // 2) * (latent_w // 2)  # 5 * 16 * 16 = 1280
    
    print(f"\nTest configuration:")
    print(f"  Video: {num_frames} frames × {height}×{width}")
    print(f"  Latent: {latent_frames} × {latent_h} × {latent_w}")
    print(f"  Sequence length: {seq_len} tokens")
    
    # Load model
    print(f"\nLoading transformer from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"  Device: {device}")
    
    # Detect format and load accordingly
    if is_official_format(model_path):
        print("  Detected: Official Wan format")
        transformer = load_transformer_from_official(model_path, dtype, device)
    else:
        print("  Detected: Diffusers format")
        transformer = load_transformer_from_diffusers(model_path, dtype, device)
    
    print("  ✓ Transformer loaded")
    
    # Create test inputs
    print("\nCreating test inputs...")
    torch.manual_seed(42)
    
    # Hidden states: [B, C, F, H, W]
    hidden_states = torch.randn(
        batch_size, 16, latent_frames, latent_h, latent_w,
        dtype=dtype, device=device
    )
    
    # Timestep: [B]
    timestep = torch.tensor([500.0], dtype=dtype, device=device)
    
    # Encoder hidden states (text): [B, 512, 4096]
    encoder_hidden_states = torch.randn(
        batch_size, 512, 4096,
        dtype=dtype, device=device
    )
    
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    
    # Run forward pass
    print("\nRunning forward pass...")
    
    results = {}
    
    # Save inputs
    results["input_hidden_states"] = hidden_states.cpu().float()
    results["input_timestep"] = timestep.cpu().float()
    results["input_encoder_hidden_states"] = encoder_hidden_states.cpu().float()
    
    with torch.no_grad():
        # Get RoPE embeddings
        rotary_emb = transformer.rope(hidden_states)
        results["rope_cos"] = rotary_emb[0].cpu().float()
        results["rope_sin"] = rotary_emb[1].cpu().float()
        print(f"  RoPE: cos={rotary_emb[0].shape}, sin={rotary_emb[1].shape}")
        
        # Patch embedding
        patch_out = transformer.patch_embedding(hidden_states)
        patch_out_flat = patch_out.flatten(2).transpose(1, 2)
        results["patch_embedding_output"] = patch_out_flat.cpu().float()
        print(f"  Patch embedding: {patch_out_flat.shape}")
        
        # Condition embedder
        temb, timestep_proj, enc_text, enc_img = transformer.condition_embedder(
            timestep, encoder_hidden_states, None, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        
        results["temb"] = temb.cpu().float()
        results["timestep_proj"] = timestep_proj.cpu().float()
        results["encoder_text_proj"] = enc_text.cpu().float()
        print(f"  temb: {temb.shape}")
        print(f"  timestep_proj: {timestep_proj.shape}")
        print(f"  encoder_text_proj: {enc_text.shape}")
        
        # Full forward pass
        output = transformer(
            hidden_states,
            timestep,
            encoder_hidden_states,
            return_dict=True,
        )
        
        results["output_sample"] = output.sample.cpu().float()
        print(f"  Output: {output.sample.shape}")
    
    # Save config info
    results["config"] = torch.tensor([
        batch_size, num_frames, height, width,
        latent_frames, latent_h, latent_w, seq_len,
        transformer.config.num_attention_heads,
        transformer.config.attention_head_dim,
        transformer.config.num_layers,
    ], dtype=torch.int64)
    
    # Save results
    output_path = "gen_wan_transformer_debug.safetensors"
    
    # Ensure all tensors are contiguous
    for key in results:
        if isinstance(results[key], torch.Tensor):
            results[key] = results[key].contiguous()
    
    save_file(results, output_path)
    print(f"\n✓ Saved {len(results)} tensors to {output_path}")
    
    # Print tensor summary
    print("\nTensor summary:")
    for key in sorted(results.keys()):
        t = results[key]
        print(f"  {key}: shape={list(t.shape)}, dtype={t.dtype}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
