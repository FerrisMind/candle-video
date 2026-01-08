import torch
import sys
import os
import numpy as np
from safetensors.torch import save_file

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

# Configuration matching default LTX-Video 0.9.5 (from src/models/ltx_video/ltx_transformer.rs)
config = {
    "num_attention_heads": 32,
    "attention_head_dim": 64,
    "in_channels": 128,
    "out_channels": 128,
    "num_layers": 28,
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 4096,
    "attention_bias": True,
    "attention_out_bias": True,
    "patch_size": 1,
    "patch_size_t": 1,
}

print("Loading DiT...")
# We use from_pretrained to get real weights, or just init with config if we want to test logic.
# Since we have weights for 0.9.5, let's load them to be sure.
model_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\snapshots\e58e28c39631af4d1468ee57a853764e11c1d37e'
transformer = LTXVideoTransformer3DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=torch.float32
)
transformer.eval()

# Create dummy inputs
batch_size = 1
num_frames = 2
height = 8
width = 8
# Latents after patchify (if patch_size=1, PT=1, it's just flattened latent dims)
latent_seq_len = num_frames * height * width
hidden_states = torch.randn(batch_size, latent_seq_len, config["in_channels"])
encoder_hidden_states = torch.randn(batch_size, 128, config["caption_channels"]) # 128 text tokens
timestep = torch.tensor([500.0], dtype=torch.float32)
encoder_attention_mask = torch.ones(batch_size, 128)

# RoPE interpolation scale
rope_interpolation_scale = (1.0, 1.0, 1.0)

print("Running DiT forward pass...")
with torch.no_grad():
    output = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        rope_interpolation_scale=rope_interpolation_scale,
        return_dict=False
    )[0]

print(f"Output shape: {output.shape}")

# Save for Rust verification
tensors = {
    "hidden_states": hidden_states,
    "encoder_hidden_states": encoder_hidden_states,
    "timestep": timestep,
    "encoder_attention_mask": encoder_attention_mask,
    "output": output,
}
# We can also capture some intermediate values via hooks if needed later.

save_file(tensors, "dit_verification.safetensors")
print("Saved verification data to dit_verification.safetensors")
