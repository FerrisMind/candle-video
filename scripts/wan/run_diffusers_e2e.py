#!/usr/bin/env python3
"""Full Wan Diffusers reference run (CUDA + F16 by default)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import add_runtime_args, load_wan_pipeline, runtime_from_args

import numpy as np
import torch
from PIL import Image

DEFAULT_ROOT = Path("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", default="A cat walking in the snow")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--latents-json",
        type=Path,
        default=None,
        help="Initial latents from scripts/wan/export_latents.py (PyTorch RNG)",
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output/wan-diffusers-ref"))
    args = parser.parse_args()

    device, dtype = runtime_from_args(args)
    print(f"Diffusers runtime: device={device}, dtype={dtype}")

    pipe = load_wan_pipeline(str(args.root), device, dtype, full=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = None
    if args.latents_json is not None:
        import json

        payload = json.loads(args.latents_json.read_text())
        shape = tuple(payload["shape"])
        latents = (
            torch.tensor(payload["data"], dtype=torch.float32)
            .reshape(shape)
            .to(device)
        )
        print(f"Using initial latents from {args.latents_json} shape={shape}")

    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        latents=latents,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = out.frames[0]
    arr = np.array(frames)
    print(f"frames shape={arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f} std={arr.std():.2f}")

    if isinstance(frames[0], np.ndarray):
        pil_frames = [
            Image.fromarray((np.clip(f, 0.0, 1.0) * 255).astype(np.uint8)) for f in frames
        ]
    else:
        pil_frames = list(frames)

    gif_path = args.output_dir / "video.gif"
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
