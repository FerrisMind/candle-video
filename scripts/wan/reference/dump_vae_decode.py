#!/usr/bin/env python3
"""Dump Wan VAE decode reference tensors for Rust parity tests."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def load_safetensors_shapes(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


def manual_decode_stub(latent: list[float], mean: list[float], std: list[float]) -> list[float]:
    """Latent denorm only (no torch/diffusers required)."""
    out = []
    c = len(mean)
    for i, v in enumerate(latent):
        out.append(v / std[i % c] + mean[i % c])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path = args.vae_dir / "config.json"
    weights_path = args.vae_dir / "diffusion_pytorch_model.safetensors"
    cfg = json.loads(config_path.read_text())
    shapes = load_safetensors_shapes(weights_path)

    # Tiny latent for a lightweight reference fixture.
    import random

    random.seed(args.seed)
    z_dim = cfg["z_dim"]
    latent = [random.gauss(0, 1) for _ in range(z_dim * 4 * 4)]
    denorm = manual_decode_stub(latent, cfg["latents_mean"], cfg["latents_std"])

    fixture: dict = {
        "z_dim": z_dim,
        "latent_shape": [1, z_dim, 1, 4, 4],
        "latent": latent,
        "denormalized_latent": denorm,
        "decoder_weight_count": sum(1 for k in shapes if k.startswith("decoder.")),
        "post_quant_keys": [k for k in shapes if k.startswith("post_quant_conv.")],
    }

    try:
        import torch
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(args.vae_dir))
        vae.eval()
        latents_mean = torch.tensor(cfg["latents_mean"]).view(1, z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(cfg["latents_std"]).view(1, z_dim, 1, 1, 1)
        z = torch.tensor(latent, dtype=torch.float32).view(1, z_dim, 1, 4, 4)
        z = z / latents_std + latents_mean
        with torch.no_grad():
            out = vae.decode(z).sample
        fixture["decoded_shape"] = list(out.shape)
        fixture["decoded"] = out.flatten().tolist()
        fixture["backend"] = "diffusers"
    except ImportError:
        fixture["backend"] = "manual_denorm_only"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2))
    print(f"Wrote {args.output} ({fixture['backend']})")


if __name__ == "__main__":
    main()
