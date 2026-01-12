#!/usr/bin/env python3
"""
Test Wan transformer memory usage at different resolutions.

This helps identify the memory requirements for different video sizes.

Supports both:
- Official Wan weights (wan2.1_t2v_1.3B_fp16.safetensors)
- Diffusers format (transformer/diffusion_pytorch_model.safetensors)

Run with:
    python scripts/test_wan_memory.py
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


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_gpu_memory_reserved():
    """Get reserved GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    return 0


def test_resolution(transformer, num_frames, height, width, device, dtype):
    """Test a single resolution and return memory usage."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Calculate dimensions
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    seq_len = latent_frames * (latent_h // 2) * (latent_w // 2)
    
    print(f"\n{'='*60}")
    print(f"Testing: {num_frames} frames × {height}×{width}")
    print(f"  Latent: {latent_frames} × {latent_h} × {latent_w}")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Memory before: {get_gpu_memory():.2f} GB")
    
    try:
        # Create inputs
        hidden_states = torch.randn(
            1, 16, latent_frames, latent_h, latent_w,
            dtype=dtype, device=device
        )
        timestep = torch.tensor([500.0], dtype=dtype, device=device)
        encoder_hidden_states = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        
        print(f"  Memory after inputs: {get_gpu_memory():.2f} GB")
        
        # Forward pass
        with torch.no_grad():
            output = transformer(
                hidden_states,
                timestep,
                encoder_hidden_states,
                return_dict=True,
            )
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  Output shape: {output.sample.shape}")
        
        # Cleanup
        del hidden_states, timestep, encoder_hidden_states, output
        gc.collect()
        torch.cuda.empty_cache()
        
        return peak_memory, seq_len
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM ERROR: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None, seq_len
    except Exception as e:
        print(f"  ERROR: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None, seq_len


def main():
    print("=" * 60)
    print("Wan Transformer Memory Test")
    print("=" * 60)
    
    # Check for model
    model_path = "models/Wan2.1-T2V-1.3B"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16  # Use FP16 for memory efficiency
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GB")
    
    # Load model
    print(f"\nLoading transformer from {model_path}...")
    
    transformer = load_transformer(model_path, dtype, device)
    
    model_memory = get_gpu_memory()
    print(f"Model loaded. Memory: {model_memory:.2f} GB")
    
    # Test configurations (increasing size)
    test_configs = [
        # (frames, height, width)
        (17, 256, 256),    # Small - should work
        (17, 384, 384),    # Medium
        (33, 384, 384),    # More frames
        (49, 480, 480),    # 480p-ish
        (81, 480, 480),    # Full 480p frames
        (81, 480, 832),    # Target resolution
    ]
    
    results = []
    
    for num_frames, height, width in test_configs:
        peak_mem, seq_len = test_resolution(
            transformer, num_frames, height, width, device, dtype
        )
        results.append({
            "frames": num_frames,
            "height": height,
            "width": width,
            "seq_len": seq_len,
            "peak_memory": peak_mem,
        })
        
        if peak_mem is None:
            print("  Stopping tests due to OOM")
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Frames':>6} {'H×W':>10} {'Seq Len':>10} {'Peak Mem':>10}")
    print("-" * 40)
    
    for r in results:
        mem_str = f"{r['peak_memory']:.2f} GB" if r['peak_memory'] else "OOM"
        print(f"{r['frames']:>6} {r['height']}×{r['width']:>4} {r['seq_len']:>10} {mem_str:>10}")
    
    print("\nNote: CFG (guidance_scale > 1) requires ~2x memory for two forward passes")


if __name__ == "__main__":
    main()
