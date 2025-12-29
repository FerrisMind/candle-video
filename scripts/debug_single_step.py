#!/usr/bin/env python3
"""
Detailed comparison: Run one DiT step in diffusers and save all intermediates.
This will help identify where Rust implementation diverges.
"""

import torch
import numpy as np
from pathlib import Path


def save_tensor(tensor: torch.Tensor, path: str, name: str):
    """Save tensor for comparison."""
    data = tensor.float().cpu().numpy()
    np.save(path, data)
    print(f"Saved {name}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")


def run_single_step_debug():
    from diffusers import LTXPipeline
    
    output_dir = Path("output/debug_step")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading diffusers pipeline...")
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    
    # Test parameters
    prompt = "A red apple on a wooden table"
    negative_prompt = "low quality"
    width = 512
    height = 320
    num_frames = 25
    num_inference_steps = 20
    guidance_scale = 3.0
    seed = 42
    
    # Set seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Get dims
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio
    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    
    print(f"Latent dims: {latent_num_frames}x{latent_height}x{latent_width}")
    
    # 1. Create text embeddings
    print("\n1. Creating text embeddings...")
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            device="cuda",
        )
    
    # CFG: concat negative + positive
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    
    save_tensor(prompt_embeds, str(output_dir / "prompt_embeds.npy"), "prompt_embeds")
    save_tensor(prompt_attention_mask, str(output_dir / "prompt_attention_mask.npy"), "prompt_attention_mask")
    
    # 2. Create initial latents
    print("\n2. Creating initial latents...")
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    
    save_tensor(latents, str(output_dir / "initial_latents.npy"), "initial_latents (packed)")
    print(f"  Packed latents shape: {latents.shape}")
    
    # 3. Setup scheduler
    print("\n3. Setting up scheduler...")
    video_sequence_length = latent_num_frames * latent_height * latent_width
    
    base_seq_len = 256
    max_seq_len = 4096
    base_shift = 0.5
    max_shift = 1.15
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = video_sequence_length * m + b
    
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, sigmas=sigmas, mu=mu, device="cuda")
    
    print(f"  mu (shift): {mu:.6f}")
    print(f"  First timestep: {pipe.scheduler.timesteps[0]}")
    print(f"  First sigma: {pipe.scheduler.sigmas[0]}")
    
    # 4. RoPE interpolation scale
    rope_interpolation_scale = (
        1 / ((pipe.transformer.config.video_seq_length / latent_num_frames) ** 0.5),
        1 / ((pipe.transformer.config.video_seq_length / latent_height) ** 0.5),
        1 / ((pipe.transformer.config.video_seq_length / latent_width) ** 0.5),
    )
    print(f"  RoPE scale: {rope_interpolation_scale}")
    
    # 5. Run ONE step
    print("\n4. Running ONE transformer step...")
    t = pipe.scheduler.timesteps[0]  # First timestep
    
    # Duplicate latents for CFG
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = latent_model_input.to(prompt_embeds.dtype)
    
    save_tensor(latent_model_input, str(output_dir / "latent_model_input.npy"), "latent_model_input (CFG)")
    
    # Expand timestep
    timestep = t.expand(latent_model_input.shape[0])
    print(f"  Timestep tensor: {timestep}")
    
    save_tensor(timestep, str(output_dir / "timestep.npy"), "timestep")
    
    # Run transformer
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            return_dict=False,
        )[0]
    
    noise_pred = noise_pred.float()
    save_tensor(noise_pred, str(output_dir / "noise_pred_raw.npy"), "noise_pred (raw, before CFG)")
    
    # 5. Apply CFG
    print("\n5. Applying CFG...")
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    save_tensor(noise_pred_guided, str(output_dir / "noise_pred_guided.npy"), "noise_pred (after CFG)")
    
    # 6. Scheduler step  
    print("\n6. Running scheduler step...")
    latents_after_step = pipe.scheduler.step(noise_pred_guided.to(torch.bfloat16), t, latents, return_dict=False)[0]
    
    save_tensor(latents_after_step, str(output_dir / "latents_after_step.npy"), "latents_after_step")
    
    print("\n" + "="*60)
    print("Debug data saved!")
    print(f"Directory: {output_dir}")
    print("\nFiles saved:")
    for f in sorted(output_dir.glob("*.npy")):
        print(f"  - {f.name}")
    print("\nCompare these with Rust implementation outputs.")


if __name__ == "__main__":
    run_single_step_debug()
