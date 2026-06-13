#!/usr/bin/env python3
"""Dump Wan UniPC scheduler sigmas/timesteps and step trajectory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/wan/scheduler_unipc_flow.json"
SCHEDULER_DIRS = [
    ROOT.parent / "Wan2.1-T2V-1.3B-Diffusers/scheduler",
    Path("/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers/scheduler"),
]


def tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().float().numpy().reshape(-1).tolist()


def main() -> int:
    sched_dir = next((p for p in SCHEDULER_DIRS if p.exists()), None)
    if sched_dir is None:
        print("scheduler config not found", file=sys.stderr)
        return 1

    sched = UniPCMultistepScheduler.from_pretrained(sched_dir)
    sched.set_timesteps(50)

    torch.manual_seed(11)
    device = torch.device("cpu")
    sample = torch.randn(1, 8, device=device, dtype=torch.float32) * 0.1
    trajectory = [tensor_to_list(sample)]

    for t in sched.timesteps[:5]:
        t_val = int(t.item())
        model_output = sample * 0.05
        out = sched.step(model_output, t_val, sample).prev_sample
        sample = out
        trajectory.append(tensor_to_list(sample))

    payload = {
        "num_inference_steps": 50,
        "sigmas": sched.sigmas.cpu().float().numpy().tolist(),
        "timesteps": sched.timesteps.cpu().numpy().tolist(),
        "trajectory_shapes": [[1, 8]] * len(trajectory),
        "trajectory": trajectory,
    }

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
