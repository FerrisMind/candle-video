import torch
import sys
import json
import numpy as np

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from safetensors import safe_open
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

# Load config
config_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\vae\config.json'
weights_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\vae\diffusion_pytorch_model.safetensors'

# Load VAE
print("Loading VAE...")
vae = AutoencoderKLLTXVideo.from_pretrained(
    r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5',
    subfolder='vae',
    torch_dtype=torch.float32,
    local_files_only=True
)
vae.eval()

# Create test latents [B=1, C=128, F=2, H=8, W=8]
# This matches the input to decoder
print("Creating test latents...")
torch.manual_seed(42)
latents = torch.randn(1, 128, 2, 8, 8, dtype=torch.float32)

print(f"Input latents shape: {latents.shape}")
print(f"Input latents first 5 (raw): {latents.flatten()[:5].tolist()}")

# Denormalize latents like the pipeline does (lines 816-817 in pipeline_ltx.py)
# latents = latents * latents_std / scaling_factor + latents_mean
latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1)
latents_std = vae.latents_std.view(1, -1, 1, 1, 1)
scaling_factor = vae.config.scaling_factor
latents = latents * latents_std / scaling_factor + latents_mean

print(f"Input latents first 5 (after denorm): {latents.flatten()[:5].tolist()}")

# Save latents for Rust comparison (DENORMALIZED latents!)
np.save('test_latents.npy', latents.numpy())
print("Saved denormalized test latents to test_latents.npy")

# Hook to capture outputs for comparison
captured_outputs = {}
def get_hook(name):
    def hook(model, input, output):
        captured_outputs[name] = output.detach().cpu()
        # Also capture input for time_embedder
        if "time_embedder" in name and input:
            captured_outputs[name + "_input"] = input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu()
    return hook

# Register hooks
vae.decoder.conv_in.register_forward_hook(get_hook("conv_in"))
vae.decoder.mid_block.register_forward_hook(get_hook("mid_block"))
vae.decoder.mid_block.time_embedder.register_forward_hook(get_hook("mid_block_time_embedder"))

# Decode
print("\nRunning VAE decode...")
with torch.no_grad():
    # VAE has timestep_conditioning=True, need to provide temb tensor
    temb = torch.tensor([0.05], dtype=torch.float32)  # typical decode_timestep
    decoded = vae.decode(latents, temb=temb, return_dict=False)[0]

# Print captured outputs for comparison
if "conv_in" in captured_outputs:
    conv_in_out = captured_outputs["conv_in"]
    np.save('conv_in_python.npy', conv_in_out.numpy())
    print(f"\nConv_in output shape: {conv_in_out.shape}")
    print(f"Conv_in output mean: {conv_in_out.mean().item():.4f}")
    print(f"Conv_in output first 5: {conv_in_out.flatten()[:5].tolist()}")
    print(f"Conv_in output min/max: {conv_in_out.min().item():.4f}/{conv_in_out.max().item():.4f}")

if "mid_block" in captured_outputs:
    mid_block_out = captured_outputs["mid_block"]
    np.save('mid_block_python.npy', mid_block_out.numpy())
    np.savetxt('mid_block_python.txt', mid_block_out.flatten().numpy(), fmt='%.6f')
    print(f"\nMid block shape: {mid_block_out.shape}")
    print(f"Mid block first 20: {mid_block_out.flatten()[:20].tolist()}")

if "mid_block_time_embedder" in captured_outputs:
    te_out = captured_outputs["mid_block_time_embedder"]
    print(f"\nTime embedder output shape: {te_out.shape}")
    print(f"Time embedder output first 10: {te_out.flatten()[:10].tolist()}")
    print(f"Time embedder output mean: {te_out.mean().item():.4f}")


print(f"Decoded shape: {decoded.shape}")
print(f"Decoded first 10 values: {decoded.flatten()[:10].tolist()}")
print(f"Decoded min: {decoded.min().item()}, max: {decoded.max().item()}")
print(f"Decoded mean: {decoded.mean().item()}")

# Save to file for Rust comparison
# Save for Rust verification
from safetensors.torch import save_file

tensors = {
    "latents": latents,
    "decoded": decoded,
    "temb": torch.tensor([0.05], dtype=torch.float32) # Save the temb used
}
save_file(tensors, "vae_verification.safetensors")
print("\nSaved latents and decoded output to vae_verification.safetensors")

