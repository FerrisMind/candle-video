import torch
import numpy as np
import sys
from safetensors.torch import save_file

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline, calculate_shift, retrieve_timesteps
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

def test_pipeline_logic():
    device = "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

    # Configs
    transformer_config = {
        "in_channels": 128,
        "out_channels": 128,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "num_layers": 1,
        "patch_size": 1,
        "patch_size_t": 1,
    }
    
    scheduler_config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": False,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "time_shift_type": "exponential",
    }

    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
    transformer = LTXVideoTransformer3DModel(**transformer_config).to(device, dtype)
    
    # Parameters
    height = 32
    width = 32
    num_frames = 9
    vae_spatial_compression_ratio = 32
    vae_temporal_compression_ratio = 8
    
    latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1
    latent_height = height // vae_spatial_compression_ratio
    latent_width = width // vae_spatial_compression_ratio
    video_sequence_length = latent_num_frames * latent_height * latent_width
    
    # Init Latents
    batch_size = 1
    num_channels_latents = 128
    shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
    latents = torch.randn(shape, device=device, dtype=dtype)
    packed_latents = LTXPipeline._pack_latents(latents, 1, 1)

    # Prompt Embeds
    max_seq_len = 128
    prompt_embeds = torch.randn(batch_size, max_seq_len, 4096, device=device, dtype=dtype)
    prompt_attention_mask = torch.ones(batch_size, max_seq_len, device=device, dtype=torch.long)
    negative_prompt_embeds = torch.randn(batch_size, max_seq_len, 4096, device=device, dtype=dtype)
    negative_prompt_attention_mask = torch.ones(batch_size, max_seq_len, device=device, dtype=torch.long)
    cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    cfg_prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # Steps
    num_inference_steps = 5
    guidance_scale = 3.0
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        video_sequence_length,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu)
    rope_interpolation_scale = (8/24, 32, 32)

    current_latents = packed_latents
    step_results = []
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([current_latents] * 2)
        ts_input = t.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=cfg_prompt_embeds,
                timestep=ts_input,
                encoder_attention_mask=cfg_prompt_attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                rope_interpolation_scale=rope_interpolation_scale,
                return_dict=False,
            )[0]
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        current_latents = scheduler.step(noise_pred_cfg, t, current_latents, return_dict=False)[0]
        step_results.append(current_latents.clone())

    # Save
    tensors = {
        "initial_latents": latents.contiguous(),
        "packed_latents": packed_latents.contiguous(),
        "prompt_embeds": prompt_embeds.contiguous(),
        "prompt_attention_mask": prompt_attention_mask.contiguous(),
        "negative_prompt_embeds": negative_prompt_embeds.contiguous(),
        "negative_prompt_attention_mask": negative_prompt_attention_mask.contiguous(),
        "timesteps": timesteps.to(torch.float32).contiguous(),
        "sigmas": scheduler.sigmas.to(torch.float32).contiguous(),
    }
    for i, res in enumerate(step_results):
        tensors[f"latents_step_{i}"] = res.contiguous()
    for name, param in transformer.named_parameters():
        tensors[f"transformer.{name}"] = param.detach().cpu().contiguous()
        
    save_file(tensors, "pipeline_verification.safetensors")
    print("Saved pipeline_verification.safetensors")

if __name__ == "__main__":
    test_pipeline_logic()
