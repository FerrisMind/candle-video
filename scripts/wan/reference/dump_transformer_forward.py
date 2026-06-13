#!/usr/bin/env python3
"""Dump full WanTransformer3DModel forward I/O for Rust parity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/wan/transformer_forward_tiny.json"
MODEL_DIRS = [
    ROOT.parent / "Wan2.1-T2V-1.3B-Diffusers/transformer",
    Path("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers/transformer"),
]


def tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().float().numpy().reshape(-1).tolist()


def main() -> int:
    model_dir = next((p for p in MODEL_DIRS if p.exists()), None)
    if model_dir is None:
        print("Wan transformer fixture not found", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import resolve_device, resolve_dtype

    torch.manual_seed(7)
    device = resolve_device()
    dtype = resolve_dtype(device)
    print(f"Diffusers runtime: device={device}, dtype={dtype}")

    model = WanTransformer3DModel.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    batch = 1
    hidden_states = torch.randn(batch, 16, 1, 4, 4, device=device, dtype=dtype) * 0.02
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    encoder_hidden_states = torch.randn(batch, 8, 4096, device=device, dtype=dtype) * 0.02

    with torch.no_grad():
        pe = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, enc_proj, _ = model.condition_embedder(
            timestep, encoder_hidden_states
        )
        out = model(hidden_states, timestep, encoder_hidden_states).sample

    payload = {
        "shapes": {
            "hidden_states": list(hidden_states.shape),
            "timestep": list(timestep.shape),
            "encoder_hidden_states": list(encoder_hidden_states.shape),
            "output": list(out.shape),
        },
        "hidden_states": tensor_to_list(hidden_states),
        "timestep": tensor_to_list(timestep.float()),
        "encoder_hidden_states": tensor_to_list(encoder_hidden_states),
        "output": tensor_to_list(out),
        "intermediates": {
            "patch_embed_row0": pe[0, 0, :16].tolist(),
            "temb_row0": temb[0, :16].tolist(),
            "enc_proj_row0": enc_proj[0, 0, :16].tolist(),
        },
    }

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
