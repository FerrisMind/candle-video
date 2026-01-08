import torch
import sys
import json
import numpy as np

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

# Load VAE
print("Loading VAE...")
vae = AutoencoderKLLTXVideo.from_pretrained(
    r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\snapshots\e58e28c39631af4d1468ee57a853764e11c1d37e',
    subfolder='vae',
    torch_dtype=torch.float32
)
vae.eval()

# Create test video input [B=1, C=3, F=9, H=64, W=64] (smaller for faster testing)
# Note: VAE expects video normalized to [-1, 1]
print("Creating test video input...")
torch.manual_seed(42)
# Shape: [B, C, F, H, W] = [1, 3, 9, 64, 64]
video_input = torch.randn(1, 3, 9, 64, 64, dtype=torch.float32)
video_input = video_input.clamp(-1, 1)  # Clamp to valid range

print(f"Input video shape: {video_input.shape}")
print(f"Input video first 5: {video_input.flatten()[:5].tolist()}")
print(f"Input video min/max: {video_input.min().item():.4f}/{video_input.max().item():.4f}")

# Hook to capture encoder output
captured_outputs = {}
def get_hook(name):
    def hook(model, input, output):
        captured_outputs[name] = output.detach().cpu()
    return hook

# Register hooks
vae.encoder.conv_in.register_forward_hook(get_hook("encoder_conv_in"))
vae.encoder.mid_block.register_forward_hook(get_hook("encoder_mid_block"))
vae.encoder.conv_out.register_forward_hook(get_hook("encoder_conv_out"))

# Encode
print("\nRunning VAE encode...")
with torch.no_grad():
    posterior = vae.encode(video_input, return_dict=False)[0]
    # Sample from posterior (or use mean for deterministic comparison)
    latents_mean = posterior.mean
    latents_std = posterior.std
    latents_sample = posterior.sample()

print(f"\nEncoder outputs:")
print(f"Latents (mean) shape: {latents_mean.shape}")
print(f"Latents (mean) first 10: {latents_mean.flatten()[:10].tolist()}")
print(f"Latents (mean) min/max: {latents_mean.min().item():.4f}/{latents_mean.max().item():.4f}")
print(f"Latents mean value: {latents_mean.mean().item():.4f}")

# Print captured outputs
if "encoder_conv_in" in captured_outputs:
    out = captured_outputs["encoder_conv_in"]
    print(f"\nEncoder conv_in output shape: {out.shape}")
    print(f"Encoder conv_in output first 5: {out.flatten()[:5].tolist()}")
    print(f"Encoder conv_in output mean: {out.mean().item():.4f}")

if "encoder_conv_out" in captured_outputs:
    out = captured_outputs["encoder_conv_out"]
    print(f"\nEncoder conv_out output shape: {out.shape}")
    print(f"Encoder conv_out output first 10: {out.flatten()[:10].tolist()}")
    print(f"Encoder conv_out output mean: {out.mean().item():.4f}")

# Save for Rust verification
# We save latents_mean for deterministic comparison (not sampled)
tensors = {
    "video_input": video_input,
    "latents_mean": latents_mean,  # [B, 2*latent_channels, F', H', W'] - mean of posterior
    "latents_logvar": posterior.logvar,  # log variance
}
save_file(tensors, "vae_encode_verification.safetensors")
print("\nSaved video_input and latents to vae_encode_verification.safetensors")

# Also save as numpy for easy inspection
np.save('encoder_latents_mean.npy', latents_mean.numpy())
print("Saved encoder_latents_mean.npy")
