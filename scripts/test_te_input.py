import torch
import sys

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

# Load VAE
print("Loading VAE...")
vae = AutoencoderKLLTXVideo.from_pretrained(
    r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5',
    subfolder='vae',
    torch_dtype=torch.float32,
    local_files_only=True
)
vae.eval()

# Hook to capture time_embedder INPUT (including kwargs)
captured = {}
def get_pre_hook(name):
    def hook(model, args, kwargs):
        captured[name + "_args"] = args
        captured[name + "_kwargs"] = kwargs
    return hook

def get_output_hook(name):
    def hook(model, input, output):
        captured[name + "_output"] = output.detach().cpu()
    return hook

vae.decoder.mid_block.time_embedder.register_forward_pre_hook(get_pre_hook("mid_block_te"), with_kwargs=True)
vae.decoder.mid_block.time_embedder.register_forward_hook(get_output_hook("mid_block_te"))

# Create test latents (same as test_vae_decode.py)
torch.manual_seed(42)
latents = torch.randn(1, 128, 2, 8, 8, dtype=torch.float32)

# Denormalize (same as test_vae_decode.py)
latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1)
latents_std = vae.latents_std.view(1, -1, 1, 1, 1)
scaling_factor = vae.config.scaling_factor
latents = latents * latents_std / scaling_factor + latents_mean

# Decode with temb=0.05
print("\nDecoding with temb=[0.05]...")
temb = torch.tensor([0.05], dtype=torch.float32)

with torch.no_grad():
    decoded = vae.decode(latents, temb=temb, return_dict=False)[0]

# Print captured data
print("\n=== Time Embedder Analysis ===")
if "mid_block_te_kwargs" in captured:
    kwargs = captured["mid_block_te_kwargs"]
    print(f"Kwargs keys: {list(kwargs.keys())}")
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, values={v.flatten()[:5].tolist()}")
        else:
            print(f"  {k}: {v}")

if "mid_block_te_output" in captured:
    out = captured["mid_block_te_output"]
    print(f"\nOutput shape: {out.shape}")
    print(f"Output first 10: {out.flatten()[:10].tolist()}")
    print(f"Output mean: {out.mean().item():.6f}")
