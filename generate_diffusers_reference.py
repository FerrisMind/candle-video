import torch
import sys
import os
import math
from safetensors import safe_open

# Add local diffusers source to path
sys.path.insert(0, os.path.join(os.getcwd(), "tp", "diffusers", "src"))

from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# --- Pcg32 Implementation ---
class Pcg32:
    def __init__(self, seed, inc):
        self.state = 0
        self.inc = (inc << 1) | 1
        self.next_u32()
        self.state = (self.state + seed) & 0xFFFFFFFFFFFFFFFF
        self.next_u32()

    def next_u32(self):
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = (oldstate >> 59)
        xorshifted = xorshifted & 0xFFFFFFFF
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next_f32(self):
        return (self.next_u32() >> 8) * 5.9604645e-8

    def next_gaussian(self):
        while True:
            x = self.next_f32()
            if x > 1e-7:
                 break
        u1 = x
        u2 = self.next_f32()

        mag = math.sqrt(-2.0 * math.log(u1))
        z0 = mag * math.cos(2.0 * math.pi * u2)
        z1 = mag * math.sin(2.0 * math.pi * u2)
        
        return z0, z1

    def randn(self, shape):
        count = 1
        for d in shape:
            count *= d
            
        data = []
        i = 0
        while i < count:
            z0, z1 = self.next_gaussian()
            data.append(z0)
            if i + 1 < count:
                data.append(z1)
            i += 2
            
        return torch.tensor(data[:count]).reshape(shape)
# -----------------------------

# Path to the model snapshot
model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
output_dir = "reference_output"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load T5 GGUF embeddings
print("Loading T5 GGUF embeddings from t5_gguf_embeddings.safetensors...")
with safe_open("t5_gguf_embeddings.safetensors", framework="pt", device=device) as f:
    prompt_embeds = f.get_tensor("prompt_embeds")
    negative_prompt_embeds = f.get_tensor("negative_prompt_embeds")
    prompt_attention_mask = f.get_tensor("prompt_attention_mask")
    negative_prompt_attention_mask = f.get_tensor("negative_prompt_attention_mask")

# Convert to bf16
prompt_embeds = prompt_embeds.to(device).to(torch.bfloat16)
negative_prompt_embeds = negative_prompt_embeds.to(device).to(torch.bfloat16)
prompt_attention_mask = prompt_attention_mask.to(device).to(torch.bfloat16)
negative_prompt_attention_mask = negative_prompt_attention_mask.to(device).to(torch.bfloat16)

print(f"Prompt embeds shape: {prompt_embeds.shape}, mean: {prompt_embeds.float().mean().item():.6f}")

print(f"Loading pipeline from {model_path}...")
pipe = LTXPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
).to(device)

# --- Generate Deterministic Latents ---
seed = 42
rng = Pcg32(seed, 1442695040888963407)

height = 512
width = 768
num_frames = 9

vae_spatial = pipe.vae_spatial_compression_ratio
vae_temporal = pipe.vae_temporal_compression_ratio

latent_height = height // vae_spatial
latent_width = width // vae_spatial
latent_frames = (num_frames - 1) // vae_temporal + 1
channels = pipe.transformer.config.in_channels

shape = (1, channels, latent_frames, latent_height, latent_width)
print(f"Generating deterministic latents with shape {shape} on CPU...")
latents_unpacked = rng.randn(shape)
# Move to device/dtype
latents_unpacked = latents_unpacked.to(device=device, dtype=pipe.transformer.dtype)

# Pack latents
print("Packing latents...")
patch_size = pipe.transformer_spatial_patch_size
patch_t = pipe.transformer_temporal_patch_size
latents_packed = LTXPipeline._pack_latents(latents_unpacked, patch_size, patch_t)
print(f"Packed shape: {latents_packed.shape}")

print(f"Generating video with GGUF T5 embeddings...")
video = pipe(
    prompt=None,  # Don't use text encoder
    negative_prompt=None,
    width=width,
    height=height,
    num_frames=num_frames, 
    num_inference_steps=30,
    guidance_scale=3.0,
    latents=latents_packed,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    output_type="np",
).frames[0]

# Save frames as images for side-by-side comparison
import PIL.Image as Image
for i, frame in enumerate(video):
    img = Image.fromarray((frame * 255).astype("uint8"))
    img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

print(f"Done! Reference frames saved to {output_dir}")

