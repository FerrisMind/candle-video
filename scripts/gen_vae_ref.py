
import torch
import numpy as np
from diffusers import AutoencoderKLLTXVideo
from safetensors.torch import save_file

def gen_vae_ref():
    print("Initializing VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Initialize with default 0.9.5 VAE config implicitly used by diffusers
    model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
    model = AutoencoderKLLTXVideo.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    # Create random latents
    # Shape based on standard config: [B, C, F, H, W]
    # In decode method: [B, C, F, H, W]
    b = 1
    c = 128
    f_latent = 8 
    h_latent = 16 
    w_latent = 16
    
    # Expected output size approximately:
    # F = (8-1)*8+1 = 57 ? No, temporal compression is 8.
    # H = 16*32 = 512
    # W = 16*32 = 512
    
    torch.manual_seed(42)
    latents = torch.randn(b, c, f_latent, h_latent, w_latent).to(device)
    # Create temb (timestep)
    # Reference config uses decode_timestep=0.05
    temb = torch.tensor([0.05], dtype=latents.dtype).expand(b).to(device)
    
    mid_block_out = None
    up_block_0_out = None
    up_block_1_out = None
    up_block_2_out = None
    conv_out_out = None
    
    def hook_mid(module, input, output):
        nonlocal mid_block_out
        mid_block_out = output.detach().cpu()
        
    def hook_up0(module, input, output):
        nonlocal up_block_0_out
        up_block_0_out = output.detach().cpu()

    def hook_up1(module, input, output):
        nonlocal up_block_1_out
        up_block_1_out = output.detach().cpu()

    def hook_up2(module, input, output):
        nonlocal up_block_2_out
        up_block_2_out = output.detach().cpu()

    def hook_conv_out(module, input, output):
        nonlocal conv_out_out
        conv_out_out = output.detach().cpu()
    
    # Register hooks
    h1 = model.decoder.mid_block.register_forward_hook(hook_mid)
    h2 = model.decoder.up_blocks[0].register_forward_hook(hook_up0)
    h3 = model.decoder.up_blocks[1].register_forward_hook(hook_up1)
    h4 = model.decoder.up_blocks[2].register_forward_hook(hook_up2)
    h5 = model.decoder.conv_out.register_forward_hook(hook_conv_out)

    print(f"Temb value: {temb}")
    print("Running decode...")
    with torch.no_grad():
        output = model.decode(latents, temb=temb, return_dict=False)[0]
    
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    
    print(f"Mid block output shape: {mid_block_out.shape}")
    print(f"UpBlock 0 output shape: {up_block_0_out.shape}")
    print(f"UpBlock 1 output shape: {up_block_1_out.shape}")
    print(f"UpBlock 2 output shape: {up_block_2_out.shape}")
    print(f"ConvOut output shape: {conv_out_out.shape}")
    print(f"Output shape: {output.shape}")

    print("Saving tensors...")
    tensors = {
        "latents": latents,
        "temb": temb,
        "output": output,
        "mid_block_out": mid_block_out,
        "up_block_0_out": up_block_0_out,
        "up_block_1_out": up_block_1_out,
        "up_block_2_out": up_block_2_out,
        "conv_out_out": conv_out_out,
    }
    
    # Save model weights to load in Rust
    model_state = model.state_dict()
    for k, v in model_state.items():
        tensors[f"vae.{k}"] = v

    save_file(tensors, "gen_vae_ref.safetensors")
    print("Done. Saved to gen_vae_ref.safetensors")

if __name__ == "__main__":
    gen_vae_ref()
