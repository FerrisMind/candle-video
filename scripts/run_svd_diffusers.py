
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import os

def run_svd():
    model_id = "models/svd"
    output_dir = "output/svd_diffusers"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading pipeline from {model_id}...")
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Failed to load from local: {e}")
        print("Trying loading with default settings (maybe it's not fp16 variant)...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        )

    if torch.cuda.is_available():
        pipe.to("cuda")
        print("Using CUDA")
    else:
        pipe.to("cpu")
        print("CUDA not available, using CPU (warning: slow)")
    # pipe.enable_model_cpu_offload() # Can be enabled if OOM
    
    # Load image
    image_path = "tp/generative-models/assets/test_image.png"
    print(f"Loading image from {image_path}...")
    image = load_image(image_path)
    image = image.resize((1024, 576))
    
    generator = torch.manual_seed(42)
    
    # Matches Rust params
    num_frames = 14
    steps = 25 # Rust defaults to 25, user ran 2 in test but let's do real run or user's requested 2? 
    # User ran with --steps 2 in the failing example. I'll use 25 for a real quality check, or 2 to be fast.
    # Let's stick to a reasonable number to check memory usage. 10 is fine.
    steps = 25 

    print(f"Running inference with {steps} steps, {num_frames} frames...")
    
    # decode_chunk_size=2 is critical for VRAM usage
    frames = pipe(
        image, 
        decode_chunk_size=2,
        generator=generator,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        num_inference_steps=steps,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        height=576,
        width=1024,
        num_frames=num_frames,
    ).frames[0]
    
    print(f"Saving {len(frames)} frames to {output_dir}...")
    for i, frame in enumerate(frames):
        frame.save(f"{output_dir}/frame_{i:04d}.png")
        
    print("Done!")

if __name__ == "__main__":
    run_svd()
