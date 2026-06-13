"""Shared Diffusers runtime defaults for Wan parity scripts."""

from __future__ import annotations

import argparse
import os
import site
from pathlib import Path


def _ensure_nvidia_shared_libs() -> None:
    """PyTorch CUDA wheels ship companion libs that are not always on the loader path."""
    paths: list[str] = []
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        if not base:
            continue
        for sub in ("nvidia/cusparselt/lib", "nvidia/cudnn/lib", "nvidia/cublas/lib"):
            p = Path(base) / sub
            if p.is_dir():
                paths.append(str(p))
    if paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths + ([existing] if existing else []))


_ensure_nvidia_shared_libs()

import torch  # noqa: E402


def resolve_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(device: torch.device, explicit: str | None = None) -> torch.dtype:
    if explicit:
        name = explicit.lower().replace("-", "")
        mapping = {
            "float32": torch.float32,
            "f32": torch.float32,
            "float16": torch.float16,
            "f16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if name not in mapping:
            raise ValueError(f"unsupported dtype {explicit!r}")
        return mapping[name]
    if device.type == "cuda":
        # Match typical Wan Diffusers inference (half precision on GPU).
        return torch.float16
    return torch.float32


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu (default: cuda when available)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="float16 | bfloat16 | float32 (default: float16 on cuda, float32 on cpu)",
    )


def runtime_from_args(args: argparse.Namespace) -> tuple[torch.device, torch.dtype]:
    device = resolve_device(getattr(args, "device", None))
    dtype = resolve_dtype(device, getattr(args, "dtype", None))
    return device, dtype


def load_wan_pipeline(root: str, device: torch.device, dtype: torch.dtype, *, full: bool = False):
    """Load WanPipeline with placement that fits 12 GB GPUs.

    When ``full`` is False (step-0 parity), only the transformer stays on ``device``;
    UMT5 runs on CPU. When ``full`` is True (e2e), use sequential CPU offload so each
    component moves to GPU only while active.
    """
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(root, torch_dtype=dtype)
    if device.type != "cuda":
        pipe.to(device)
        return pipe

    if full:
        pipe.enable_sequential_cpu_offload(gpu_id=device.index or 0)
        return pipe

    pipe.text_encoder.to("cpu")
    pipe.transformer.to(device)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to("cpu")
    return pipe


def text_encoder_device(compute_device: torch.device) -> torch.device:
    """UMT5-XXL does not fit alongside the transformer on 12 GB cards."""
    if compute_device.type == "cuda":
        return torch.device("cpu")
    return compute_device
