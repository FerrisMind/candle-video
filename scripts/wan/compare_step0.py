#!/usr/bin/env python3
"""Dump step-0 noise_pred and latents for Rust parity comparison."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import add_runtime_args, load_wan_pipeline, runtime_from_args, text_encoder_device

import torch
from diffusers.utils.torch_utils import randn_tensor

ROOT = Path("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers")
OUT = Path("/home/mod479711/Downloads/candle-video/tests/fixtures/wan/step0_reference.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    device, dtype = runtime_from_args(args)
    print(f"Diffusers runtime: device={device}, dtype={dtype}")

    enc_device = text_encoder_device(device)
    # Parity fixture: F32 transformer math (matches Rust `wan_step0_parity` and `video-generate`).
    parity_dtype = torch.float32
    pipe = load_wan_pipeline(str(args.root), device, parity_dtype, full=False)
    print(f"Text encoder on {enc_device}, transformer on {device} dtype={parity_dtype}")

    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt="A cat walking in the snow",
        negative_prompt="",
        do_classifier_free_guidance=True,
        device=enc_device,
    )
    # Store fixtures in F32 for Rust parity tests (model may run in F16/BF16).
    prompt_embeds = prompt_embeds.to(torch.float32)
    neg_embeds = neg_embeds.to(torch.float32)

    g = torch.Generator(device=device).manual_seed(42)
    latents = randn_tensor((1, 16, 3, 32, 48), generator=g, device=device, dtype=torch.float32)

    pipe.scheduler.set_timesteps(10, device=device)
    t = pipe.scheduler.timesteps[0]
    timestep = t.expand(latents.shape[0])

    latent_model_input = latents.to(device=device, dtype=pipe.transformer.dtype)
    with torch.no_grad():
        noise_cond = pipe.transformer(
            latent_model_input, timestep, prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
            return_dict=False,
        )[0].float()
        noise_uncond = pipe.transformer(
            latent_model_input, timestep, neg_embeds.to(device=device, dtype=pipe.transformer.dtype),
            return_dict=False,
        )[0].float()
    noise_pred = noise_uncond + 5.0 * (noise_cond - noise_uncond)
    latents_after = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    def stats(x: torch.Tensor) -> dict:
        return {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "shape": list(x.shape),
        }

    fixture = {
        "runtime": {
            "device": str(device),
            "dtype": str(parity_dtype).replace("torch.", ""),
            "text_encoder_device": str(enc_device),
            "note": "F32 transformer forward for Rust step0 parity; e2e uses F16 via run_diffusers_e2e.py",
        },
        "timestep": int(t),
        "latents_init": latents.flatten().tolist(),
        "latents_init_shape": list(latents.shape),
        "prompt_embeds": prompt_embeds.flatten().tolist(),
        "prompt_embeds_shape": list(prompt_embeds.shape),
        "neg_embeds": neg_embeds.flatten().tolist(),
        "neg_embeds_shape": list(neg_embeds.shape),
        "noise_cond_stats": stats(noise_cond),
        "noise_uncond_stats": stats(noise_uncond),
        "noise_cond": noise_cond.flatten().tolist(),
        "noise_uncond": noise_uncond.flatten().tolist(),
        "noise_pred_stats": stats(noise_pred),
        "latents_after_stats": stats(latents_after),
        "noise_pred_first16": noise_pred.flatten()[:16].tolist(),
        "noise_pred": noise_pred.flatten().tolist(),
        "latents_after_first16": latents_after.flatten()[:16].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2))
    print(f"Wrote {args.out}")
    print("noise_pred", fixture["noise_pred_stats"])
    print("latents_after", fixture["latents_after_stats"])


if __name__ == "__main__":
    main()
