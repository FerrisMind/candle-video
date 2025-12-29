#!/usr/bin/env python3
"""
Debug script: Run diffusers inference with LOCAL model, save latents for Rust VAE testing.

This allows isolating whether the issue is in:
1. DiT (denoising) - if diffusers latents work with our VAE
2. VAE decoder - if diffusers latents also produce grid pattern with our VAE
"""

import torch
import numpy as np
from pathlib import Path
import argparse

def save_latents_bin(latents: torch.Tensor, path: str):
    """Save latents in format compatible with Rust loader."""
    latents = latents.float().cpu().numpy()
    shape = latents.shape
    
    with open(path, 'wb') as f:
        # Write header: ndims, then each dim
        f.write(np.array([len(shape)], dtype=np.uint64).tobytes())
        f.write(np.array(shape, dtype=np.uint64).tobytes())
        # Write data
        f.write(latents.tobytes())
    
    print(f"Saved latents to {path}")
    print(f"  Shape: {shape}")
    print(f"  Dtype: float32")
    print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")


def run_diffusers_and_save_latents(model_path: str):
    """Run diffusers inference with local model and save latents at various stages."""
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video
    
    print(f"Loading diffusers pipeline from: {model_path}")
    pipe = LTXPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    
    # Test parameters (small for quick testing)
    prompt = "A red apple on a wooden table, cinematic lighting"
    negative_prompt = "low quality, blurry"
    width = 512
    height = 320
    num_frames = 25
    num_inference_steps = 20
    guidance_scale = 3.0
    
    print(f"\nRunning inference:")
    print(f"  Prompt: {prompt}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")
    
    # Run pipeline with output_type="latent" to get raw latents
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="latent",  # Get latents instead of video
    )
    
    latents = result.frames  # This is actually latents when output_type="latent"
    print(f"\nRaw latents from pipeline:")
    print(f"  Shape: {latents.shape}")
    print(f"  Dtype: {latents.dtype}")
    
    # Save raw latents (before unpack/denormalize)
    output_dir = Path("output/debug_latents")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_latents_bin(latents, str(output_dir / "latents_raw.bin"))
    
    # Unpack latents (diffusers does this internally)
    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio
    
    latents_unpacked = pipe._unpack_latents(
        latents,
        latent_num_frames,
        latent_height,
        latent_width,
        pipe.transformer_spatial_patch_size,
        pipe.transformer_temporal_patch_size,
    )
    print(f"\nUnpacked latents:")
    print(f"  Shape: {latents_unpacked.shape}")
    
    save_latents_bin(latents_unpacked, str(output_dir / "latents_unpacked.bin"))
    
    # Denormalize latents
    latents_denorm = pipe._denormalize_latents(
        latents_unpacked,
        pipe.vae.latents_mean,
        pipe.vae.latents_std,
        pipe.vae.config.scaling_factor,
    )
    print(f"\nDenormalized latents:")
    print(f"  Shape: {latents_denorm.shape}")
    print(f"  Range: [{latents_denorm.min().item():.4f}, {latents_denorm.max().item():.4f}]")
    
    save_latents_bin(latents_denorm, str(output_dir / "latents_denormalized.bin"))
    
    # Also save with diffusers VAE decode for comparison
    print("\nDecoding with diffusers VAE...")
    latents_for_vae = latents_denorm.to(pipe.vae.dtype)
    
    # Check if VAE needs timestep conditioning
    decode_timestep = None
    if pipe.vae.config.timestep_conditioning:
        decode_timestep = torch.tensor([0.05], device=latents_for_vae.device, dtype=latents_for_vae.dtype)
    
    with torch.no_grad():
        video = pipe.vae.decode(latents_for_vae, decode_timestep, return_dict=False)[0]
    
    print(f"Decoded video shape: {video.shape}")
    
    # Postprocess and save
    video = pipe.video_processor.postprocess_video(video, output_type="pil")
    export_to_video(video, str(output_dir / "diffusers_output.mp4"), fps=24)
    print(f"\nSaved diffusers output to {output_dir / 'diffusers_output.mp4'}")
    
    # Save first and last frames as PNG
    video[0].save(str(output_dir / "diffusers_frame_000.png"))
    video[-1].save(str(output_dir / f"diffusers_frame_{len(video)-1:03d}.png"))
    
    print("\n" + "="*60)
    print("Debug latents saved!")
    print(f"Directory: {output_dir}")
    print("\nFiles:")
    print("  - latents_raw.bin: Raw packed latents from transformer")
    print("  - latents_unpacked.bin: Unpacked to [B,C,T,H,W]")
    print("  - latents_denormalized.bin: Ready for VAE decode")
    print("  - diffusers_output.mp4: Reference video from diffusers")
    print("  - diffusers_frame_*.png: First and last frames")
    print("\nTo test with Rust VAE, run:")
    print(f"  .\\target\\release\\vae-test.exe {output_dir}/latents_denormalized.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate debug latents using diffusers")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="Lightricks/LTX-Video",  # Uses HuggingFace cache if available
        help="Path to LTX-Video model (HuggingFace ID or local path)"
    )
    args = parser.parse_args()
    
    run_diffusers_and_save_latents(args.model_path)

