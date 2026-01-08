import torch
import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

def save_video_frames(video_tensor, prefix, output_dir="roundtrip_output"):
    """
    Saves frames of a [B, C, F, H, W] tensor as images.
    Expects values in [-1, 1].
    """
    os.makedirs(output_dir, exist_ok=True)
    video_tensor = ((video_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    video_tensor = video_tensor[0] # [C, F, H, W]
    num_frames = video_tensor.shape[1]
    
    for f in range(num_frames):
        frame = video_tensor[:, f, :, :].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"{prefix}_frame_{f:03d}.png"))
    print(f"Saved {num_frames} frames to {output_dir}")

# Load VAE
print("Loading VAE...")
vae = AutoencoderKLLTXVideo.from_pretrained(
    r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\snapshots\e58e28c39631af4d1468ee57a853764e11c1d37e',
    subfolder='vae',
    torch_dtype=torch.float32
)
vae.eval()

# Create test video input [B=1, C=3, F=9, H=256, W=256]
print("Creating test video input...")
torch.manual_seed(42)
video_input = torch.randn(1, 3, 9, 256, 256, dtype=torch.float32).clamp(-1, 1)

# Full Roundtrip
print("Running Roundtrip: Input -> Encoder -> Decoder -> Output")
with torch.no_grad():
    # 1. Encode
    posterior = vae.encode(video_input, return_dict=False)[0]
    latents = posterior.mode() # Use mode for deterministic test
    print(f"Latents shape: {latents.shape}")
    
    # 2. Decode
    # We need a timestep for the decoder in LTX 0.9.5
    # Let's use 0.0 (equivalent to clean sample)
    temb = torch.tensor([0.0], dtype=torch.float32)
    video_output = vae.decode(latents, temb=temb, return_dict=False)[0]
    print(f"Output video shape: {video_output.shape}")

# Comparison
mse = torch.nn.functional.mse_loss(video_input, video_output)
psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
print(f"\nRoundtrip MSE: {mse.item():.6f}")
print(f"Estimated PSNR: {psnr.item():.2f} dB")

# Save for Rust verification
from safetensors.torch import save_file
tensors = {
    "video_input": video_input,
    "latents": latents,
    "video_output": video_output,
    "temb": temb,
}
save_file(tensors, "vae_roundtrip_verification.safetensors")
print("\nSaved roundtrip data to vae_roundtrip_verification.safetensors")

print("\nRoundtrip test complete. Check 'roundtrip_output' directory for frames.")
