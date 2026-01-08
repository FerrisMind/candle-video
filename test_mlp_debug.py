import torch
import sys
import numpy as np

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from diffusers.models.embeddings import get_timestep_embedding

# Load VAE
print("Loading VAE...")
vae = AutoencoderKLLTXVideo.from_pretrained(
    r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5',
    subfolder='vae',
    torch_dtype=torch.float32,
    local_files_only=True
)
vae.eval()

# Get sinusoidal embedding
timesteps = torch.tensor([0.05], dtype=torch.float32)
sinusoidal_emb = get_timestep_embedding(
    timesteps,
    embedding_dim=256,
    flip_sin_to_cos=True,
    downscale_freq_shift=0,
    scale=1.0,
    max_period=10000
)
print(f"Sinusoidal embedding shape: {sinusoidal_emb.shape}")
print(f"Sinusoidal embedding first 5: {sinusoidal_emb.flatten()[:5].tolist()}")

# Get mid_block.time_embedder.timestep_embedder.linear_1
linear_1 = vae.decoder.mid_block.time_embedder.timestep_embedder.linear_1
print(f"\nlinear_1 weight shape: {linear_1.weight.shape}")
print(f"linear_1 weight first 5: {linear_1.weight.flatten()[:5].tolist()}")

# Forward through linear_1
with torch.no_grad():
    linear_1_output = linear_1(sinusoidal_emb)
print(f"\nlinear_1 output shape: {linear_1_output.shape}")
print(f"linear_1 output first 5: {linear_1_output.flatten()[:5].tolist()}")

# Apply silu
silu_output = torch.nn.functional.silu(linear_1_output)
print(f"\nSiLU output first 5: {silu_output.flatten()[:5].tolist()}")

# Get linear_2
linear_2 = vae.decoder.mid_block.time_embedder.timestep_embedder.linear_2
linear_2_output = linear_2(silu_output)
print(f"\nlinear_2 output (final time embedding) first 10: {linear_2_output.flatten()[:10].tolist()}")
print(f"linear_2 output mean: {linear_2_output.mean().item():.6f}")
