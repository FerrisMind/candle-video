#!/usr/bin/env python3
"""
Generate reference data for Wan transformer parity testing.

This script runs the diffusers Wan transformer with known inputs
and saves the outputs for comparison with the Rust implementation.

Usage:
    python scripts/gen_wan_transformer_ref.py

Output: gen_wan_transformer_ref.safetensors
"""

import os
import sys
import torch
from safetensors.torch import save_file

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)


def main():
    print("=" * 60)
    print("Generating Wan Transformer Reference Data")
    print("=" * 60)

    # Check for model
    model_path = "models/Wan2.1-T2V-1.3B"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Load transformer
    print(f"\nLoading transformer from {model_path}...")

    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.eval()
    transformer = transformer.to(device)
    print("✓ Transformer loaded")

    results = {}

    # Test configuration 1: Small (17 frames, 256x256)
    print("\n--- Test 1: Small (17 frames, 256x256) ---")
    results.update(run_test(transformer, device, dtype, "small", 17, 256, 256))

    # Test configuration 2: Medium (33 frames, 384x384)
    print("\n--- Test 2: Medium (33 frames, 384x384) ---")
    results.update(run_test(transformer, device, dtype, "medium", 33, 384, 384))

    # Save results
    output_path = "gen_wan_transformer_ref.safetensors"

    # Ensure all tensors are contiguous and on CPU
    for key in results:
        if isinstance(results[key], torch.Tensor):
            results[key] = results[key].cpu().float().contiguous()

    save_file(results, output_path)
    print(f"\n✓ Saved {len(results)} tensors to {output_path}")

    # Print summary
    print("\nTensor summary:")
    for key in sorted(results.keys()):
        t = results[key]
        print(f"  {key}: shape={list(t.shape)}")


def run_test(transformer, device, dtype, prefix, num_frames, height, width):
    """Run a single test configuration and return results."""
    results = {}

    # Calculate dimensions
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    seq_len = latent_frames * (latent_h // 2) * (latent_w // 2)

    print(f"  Video: {num_frames} frames × {height}×{width}")
    print(f"  Latent: {latent_frames} × {latent_h} × {latent_w}")
    print(f"  Sequence length: {seq_len} tokens")

    # Create deterministic inputs
    torch.manual_seed(42)

    hidden_states = torch.randn(
        1, 16, latent_frames, latent_h, latent_w,
        dtype=dtype, device=device
    )

    timestep = torch.tensor([500.0], dtype=dtype, device=device)

    encoder_hidden_states = torch.randn(
        1, 512, 4096,
        dtype=dtype, device=device
    )

    # Save inputs
    results[f"{prefix}_input_hidden_states"] = hidden_states.clone()
    results[f"{prefix}_input_timestep"] = timestep.clone()
    results[f"{prefix}_input_encoder_hidden_states"] = encoder_hidden_states.clone()

    # Run forward pass
    print("  Running forward pass...")

    with torch.no_grad():
        output = transformer(
            hidden_states,
            timestep,
            encoder_hidden_states,
            return_dict=True,
        )

    results[f"{prefix}_output"] = output.sample.clone()

    print(f"  Output shape: {output.sample.shape}")
    print(f"  Output mean: {output.sample.float().mean().item():.6f}")
    print(f"  Output std: {output.sample.float().std().item():.6f}")

    # Save config
    results[f"{prefix}_config"] = torch.tensor([
        num_frames, height, width,
        latent_frames, latent_h, latent_w, seq_len,
    ], dtype=torch.int64)

    return results


if __name__ == "__main__":
    main()
