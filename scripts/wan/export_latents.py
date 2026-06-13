#!/usr/bin/env python3
"""Export PyTorch initial latents for Rust/Diffusers parity (same seed → same noise)."""

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import add_runtime_args, runtime_from_args

DEFAULT_ROOT = Path("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers")


def latent_shape(height: int, width: int, num_frames: int) -> tuple[int, ...]:
    temporal = 4
    spatial = 8
    t = (num_frames - 1) // temporal + 1
    return (1, 16, t, height // spatial, width // spatial)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device, _dtype = runtime_from_args(args)
    shape = latent_shape(args.height, args.width, args.num_frames)
    g = torch.Generator(device=device).manual_seed(args.seed)
    latents = randn_tensor(shape, generator=g, device=device, dtype=torch.float32).cpu()

    payload = {
        "seed": args.seed,
        "shape": list(shape),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "data": latents.flatten().tolist(),
        "stats": {
            "min": float(latents.min()),
            "max": float(latents.max()),
            "mean": float(latents.mean()),
            "std": float(latents.std()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f"Wrote {args.output} shape={shape} stats={payload['stats']}")


if __name__ == "__main__":
    main()
