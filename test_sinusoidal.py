import torch
import sys
import numpy as np

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.models.embeddings import get_timestep_embedding

# Test with temb=0.05 and embedding_dim=256
timesteps = torch.tensor([0.05], dtype=torch.float32)

# Python PixArtAlphaCombinedTimestepSizeEmbeddings uses:
# flip_sin_to_cos=True, downscale_freq_shift=0
emb = get_timestep_embedding(
    timesteps,
    embedding_dim=256,
    flip_sin_to_cos=True,
    downscale_freq_shift=0,
    scale=1.0,
    max_period=10000
)

print(f"Python sinusoidal embedding shape: {emb.shape}")
print(f"Python sinusoidal embedding first 10: {emb.flatten()[:10].tolist()}")
print(f"Python sinusoidal embedding last 10: {emb.flatten()[-10:].tolist()}")
print(f"Python sinusoidal embedding mean: {emb.mean().item():.6f}")

np.save('sinusoidal_embedding_python.npy', emb.numpy())
print("Saved to sinusoidal_embedding_python.npy")
