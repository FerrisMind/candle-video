#!/usr/bin/env python3
"""Dump Wan transformer block 0 I/O for Rust parity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import WanTransformer3DModel

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/wan/transformer_block0.json"
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

    torch.manual_seed(42)
    device = resolve_device()
    dtype = resolve_dtype(device)
    print(f"Diffusers runtime: device={device}, dtype={dtype}")

    model = WanTransformer3DModel.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    block = model.blocks[0]

    batch = 1
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    seq = 4  # 1x2x2 patches on 1x4x4 latent
    text_len = 8

    hidden_states = torch.randn(batch, seq, dim, device=device, dtype=dtype) * 0.02
    encoder_hidden_states = torch.randn(batch, text_len, dim, device=device, dtype=dtype) * 0.02
    timestep_proj = torch.randn(batch, 6, dim, device=device, dtype=dtype) * 0.01

    latent = torch.randn(batch, model.config.in_channels, 1, 4, 4, device=device, dtype=dtype) * 0.02
    with torch.no_grad():
        rotary_emb = model.rope(latent)
        out = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    payload = {
        "config": {
            "num_attention_heads": model.config.num_attention_heads,
            "attention_head_dim": model.config.attention_head_dim,
            "ffn_dim": model.config.ffn_dim,
            "eps": model.config.eps,
            "cross_attn_norm": model.config.cross_attn_norm,
        },
        "shapes": {
            "hidden_states": list(hidden_states.shape),
            "encoder_hidden_states": list(encoder_hidden_states.shape),
            "timestep_proj": list(timestep_proj.shape),
            "latent_for_rope": list(latent.shape),
            "output": list(out.shape),
        },
        "hidden_states": tensor_to_list(hidden_states),
        "encoder_hidden_states": tensor_to_list(encoder_hidden_states),
        "timestep_proj": tensor_to_list(timestep_proj),
        "latent_for_rope": tensor_to_list(latent),
        "output": tensor_to_list(out),
    }

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
