
import torch
import numpy as np
from diffusers import LTXVideoTransformer3DModel
from safetensors.torch import save_file

def gen_ref():
    print("Initializing transformer...")
    # Use small config for verification to be fast
    print("Initializing transformer...")
    # Use small config for verification to be fast
    model = LTXVideoTransformer3DModel(
        in_channels=32,
        out_channels=32,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=16,
        cross_attention_dim=32,
        num_layers=2,
        caption_channels=32,
        qk_norm="rms_norm_across_heads",
    )
    model.eval()

    # Create random inputs
    b = 1
    f = 8
    h = 32
    w = 32
    c = 32
    seq_len = f * h * w

    torch.manual_seed(42)
    hidden_states = torch.randn(b, seq_len, c)
    encoder_hidden_states = torch.randn(b, 10, 32) # caption_channels
    timestep = torch.tensor([500.0]).repeat(b)
    # encoder_attention_mask = torch.ones(b, 10) 
    
    # LTX usually expects [B, S, C] inputs.
    # Check forward signature:
    # forward(hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, ...)
    
    # We also need rope parameters passed correctly if not standard?
    # model.forward handles rope internally or expected pre-computed?
    # Diffusers implementation computes rope inside forward usually or expects it?
    # LTXVideoTransformer3DModel in diffusers:
    # forward(hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, num_frames, height, width, ...)
    
    print("Running forward...")
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=None,
            num_frames=f,
            height=h,
            width=w,
            rope_interpolation_scale=(1.0, 1.0, 1.0),
            return_dict=False
        )[0]

    print("Saving tensors...")
    tensors = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "output": output,
    }
    
    # Save model weights too so we can load them in Rust
    # Prefix with 'transformer.' so we can use existing loading if needed or just load directly
    model_state = model.state_dict()
    for k, v in model_state.items():
        tensors[f"model.{k}"] = v

    save_file(tensors, "gen_dit_ref.safetensors")
    print("Done. Saved to gen_dit_ref.safetensors")

if __name__ == "__main__":
    gen_ref()
