#!/usr/bin/env python3
"""
Run full diffusers inference and save final denormalized latents.
Then we can compare with Rust output.
"""
import torch
import numpy as np
from pathlib import Path


def run_full_inference():
    from diffusers import LTXPipeline
    
    output_dir = Path("output/diffusers_ref")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading diffusers pipeline...")
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    
    prompt = "A red apple on a wooden table"
    negative_prompt = "low quality"
    width = 512
    height = 320
    num_frames = 25
    num_inference_steps = 20
    guidance_scale = 3.0
    seed = 42
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"Generating video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
    
    # Run inference and capture final latents
    # Use callback to capture latents at each step
    latents_history = []
    
    def callback(pipe, step, timestep, callback_kwargs):
        latents_history.append({
            'step': step,
            'timestep': timestep.item() if hasattr(timestep, 'item') else timestep,
            'latents': callback_kwargs['latents'].clone().cpu(),
        })
        return callback_kwargs
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=['latents'],
    )
    
    final_latents = result.frames  # These should be packed latents
    
    print(f"\nFinal latents shape: {final_latents.shape}")
    print(f"Final latents range: [{final_latents.min():.4f}, {final_latents.max():.4f}]")
    
    # Save raw latents (still packed 3D)
    np.save(str(output_dir / "final_latents_packed.npy"), final_latents.float().cpu().numpy())
    
    # Unpack to 5D for VAE
    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio
    
    unpacked = pipe._unpack_latents(
        final_latents,
        latent_num_frames,
        latent_height,
        latent_width,
        pipe.transformer_spatial_patch_size,
        pipe.transformer_temporal_patch_size,
    )
    
    print(f"Unpacked latents shape: {unpacked.shape}")
    print(f"Unpacked latents range: [{unpacked.min():.4f}, {unpacked.max():.4f}]")
    
    # Denormalize using VAE latent stats
    latents_mean = pipe.vae.config.get("latents_mean")
    latents_std = pipe.vae.config.get("latents_std")
    
    if latents_mean is not None and latents_std is not None:
        latents_mean = torch.tensor(latents_mean, device=unpacked.device, dtype=unpacked.dtype)
        latents_std = torch.tensor(latents_std, device=unpacked.device, dtype=unpacked.dtype)
        # Reshape for broadcasting: (C,) -> (1, C, 1, 1, 1)
        latents_mean = latents_mean.view(1, -1, 1, 1, 1)
        latents_std = latents_std.view(1, -1, 1, 1, 1)
        denormalized = unpacked * latents_std + latents_mean
    else:
        denormalized = unpacked  # No denormalization
    
    print(f"Denormalized latents shape: {denormalized.shape}")
    print(f"Denormalized latents range: [{denormalized.min():.4f}, {denormalized.max():.4f}]")
    
    # Save denormalized latents (5D)
    np.save(str(output_dir / "final_latents_denorm.npy"), denormalized.float().cpu().numpy())
    
    # Now decode with VAE
    print("\nDecoding with VAE...")
    video = pipe.vae.decode(
        denormalized.to(pipe.vae.dtype).to(pipe.vae.device),
        return_dict=False,
    )[0]
    
    print(f"Video shape: {video.shape}")
    
    # Save first frame
    from PIL import Image
    video_np = video.float().cpu().numpy()
    video_np = (video_np + 1) / 2 * 255
    video_np = video_np.clip(0, 255).astype(np.uint8)
    
    # video: [B, C, T, H, W]
    frame = video_np[0, :, 0, ...]  # first frame: [C, H, W]
    frame = frame.transpose(1, 2, 0)  # [H, W, C]
    
    Image.fromarray(frame).save(str(output_dir / "diffusers_frame_000.png"))
    Image.fromarray(video_np[0, :, -1, ...].transpose(1, 2, 0)).save(str(output_dir / "diffusers_frame_last.png"))
    
    print(f"\nSaved reference frames to {output_dir}")
    print("Compare these with Rust output to identify differences.")
    
    # Print latents history summary
    print(f"\nLatents history ({len(latents_history)} steps):")
    for h in latents_history[:3]:
        lat = h['latents']
        print(f"  Step {h['step']}: timestep={h['timestep']:.1f}, shape={lat.shape}, range=[{lat.min():.4f}, {lat.max():.4f}]")
    print("  ...")


if __name__ == "__main__":
    run_full_inference()
