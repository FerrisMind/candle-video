#!/usr/bin/env python3
"""
Generate reference latents from diffusers LTX-Video pipeline.
Uses local T5 encoder from T5-XXL-8bit directory.
"""
import torch
import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("Diffusers LTX-Video Reference Latents Generation")
    print("=" * 60)
    
    from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel, LTXPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, AutoTokenizer
    
    # Parameters matching Rust test
    model_dir = Path("ltxv-diffusers-0.9.5")
    t5_dir = Path("ltxv-2b-0.9.8-distilled/T5-XXL-8bit")
    width = 768
    height = 512
    num_frames = 25
    num_inference_steps = 30  # Matching first_pass from config
    guidance_scale = 3.0  # Distilled model uses guidance_scale=1
    seed = 284607720456150
    prompt = "A red apple on a wooden table"
    negative_prompt = "low quality"
    
    print(f"\nParameters:")
    print(f"  transformer: {model_dir}/transformer")
    print(f"  vae: {model_dir}/vae")
    print(f"  text_encoder: {t5_dir}")
    print(f"  size: {width}x{height}x{num_frames}")
    print(f"  steps: {num_inference_steps}")
    print(f"  guidance: {guidance_scale}")
    print(f"  seed: {seed}")
    print(f"  prompt: {prompt}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    
    # Load components
    print("\nLoading transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        model_dir / "transformer",
        torch_dtype=torch.bfloat16,
        variant="bf16",
    )
    
    print("Loading VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_dir / "vae",
        torch_dtype=torch.bfloat16,
    )
    
    print("Loading T5 encoder from local directory...")
    # Use AutoTokenizer for tokenizer.json support and convert path to string
    tokenizer = AutoTokenizer.from_pretrained(str(t5_dir), model_max_length=128)
    text_encoder = T5EncoderModel.from_pretrained(
        str(t5_dir),
        torch_dtype=torch.bfloat16,
    )
    
    print("Creating scheduler...")
    # LTX-Video typically requires dynamic shifting
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        shift_terminal=0.1,
        shift=1.0,
    )
    
    # Build pipeline
    print("Assembling pipeline...")
    pipe = LTXPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipe.to(device)
    
    # Generate
    print("\nGenerating...")
    generator = torch.Generator(device).manual_seed(seed)
    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
    )
    
    # Get final latents
    latents = output.frames
    
    print(f"\nFinal latents:")
    print(f"  shape: {latents.shape}")
    print(f"  dtype: {latents.dtype}")
    print(f"  range: [{latents.float().min():.4f}, {latents.float().max():.4f}]")
    
    # Save outputs
    out_dir = Path("output/diffusers_ref")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "final_latents.npy", latents.float().cpu().numpy())
    
    print(f"\nSaved to {out_dir}/final_latents.npy")
    
    # Decode and save first frame for visual comparison
    print("\nDecoding...")
    pipe.vae.to(torch.float32)
    
    # Unpack latents (3D -> 5D)
    lat_t = (num_frames - 1) // 8 + 1
    lat_h = height // 32
    lat_w = width // 32
    latent_channels = 128
    
    if latents.dim() == 3:
        latents_5d = latents.permute(0, 2, 1).reshape(1, latent_channels, lat_t, lat_h, lat_w)
    else:
        latents_5d = latents
    
    frames = pipe.vae.decode(latents_5d.float(), return_dict=False)[0]
    frames = (frames + 1) / 2
    frames = frames.clamp(0, 1)
    
    # Save first frame
    from PIL import Image
    first_frame = frames[0, :, 0].permute(1, 2, 0).detach().cpu().numpy()
    first_frame = (first_frame * 255).astype(np.uint8)
    Image.fromarray(first_frame).save(out_dir / "first_frame_diffusers.png")
    
    print(f"Saved first frame to {out_dir}/first_frame_diffusers.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
