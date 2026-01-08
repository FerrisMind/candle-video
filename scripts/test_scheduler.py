import torch
import sys
from safetensors.torch import save_file

sys.path.insert(0, r'c:\candle-video\tp\diffusers\src')
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

config = {
    "num_train_timesteps": 1000,
    "shift": 2.0,
    "use_dynamic_shifting": False,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "time_shift_type": "exponential",
}

print("Initializing Scheduler...")
scheduler = FlowMatchEulerDiscreteScheduler(**config)

num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps, device="cpu")

timesteps = scheduler.timesteps
sigmas = scheduler.sigmas

print(f"Timesteps (first 5): {timesteps[:5]}")
print(f"Sigmas (first 5): {sigmas[:5]}")

# Test step()
batch_size = 1
channels = 128
seq_len = 128
sample = torch.randn(batch_size, seq_len, channels)
model_output = torch.randn(batch_size, seq_len, channels)
timestep = timesteps[0]

prev_sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]

# Save results
tensors = {
    "timesteps": timesteps.to(torch.float32),
    "sigmas": sigmas.to(torch.float32),
    "sample_in": sample,
    "model_output": model_output,
    "sample_out": prev_sample,
}

save_file(tensors, "scheduler_verification.safetensors")
print("Saved verification data to scheduler_verification.safetensors")
