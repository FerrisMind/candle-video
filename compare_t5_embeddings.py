"""
Compare T5 GGUF embeddings (from Rust) with full precision T5 embeddings.
Identify where the embeddings diverge.
"""
import torch
import numpy as np
from safetensors import safe_open
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "tp", "diffusers", "src"))
from transformers import T5EncoderModel, AutoTokenizer

model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
text_encoder_path = os.path.join(model_path, "text_encoder")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GGUF embeddings
print("Loading GGUF embeddings...")
with safe_open("t5_gguf_embeddings.safetensors", framework="pt", device="cpu") as f:
    gguf_embeds = f.get_tensor("prompt_embeds").float()

print(f"GGUF shape: {gguf_embeds.shape}")
print(f"GGUF first 10: {gguf_embeds.flatten()[:10].tolist()}")
print(f"GGUF mean: {gguf_embeds.mean().item():.6f}")
print(f"GGUF std: {gguf_embeds.std().item():.6f}")

# Load full precision T5
print("\nLoading full precision T5...")
tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
model = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.float32).to(device)
model.eval()

prompt = "The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon."

inputs = tokenizer(prompt, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    full_embeds = outputs.last_hidden_state.cpu().float()

print(f"Full T5 shape: {full_embeds.shape}")
print(f"Full T5 first 10: {full_embeds.flatten()[:10].tolist()}")
print(f"Full T5 mean: {full_embeds.mean().item():.6f}")
print(f"Full T5 std: {full_embeds.std().item():.6f}")

# Compare
print("\n=== Comparison ===")
diff = (gguf_embeds - full_embeds).abs()
max_val, max_idx = torch.max(diff.flatten(), 0)
token_idx = (max_idx // full_embeds.shape[-1]).item()
dim_idx = (max_idx % full_embeds.shape[-1]).item()

print(f"Max diff: {max_val.item():.6f} at token {token_idx}, dim {dim_idx}")
print(f"  GGUF value: {gguf_embeds[0, token_idx, dim_idx].item():.6f}")
print(f"  Full value: {full_embeds[0, token_idx, dim_idx].item():.6f}")
print(f"Mean diff: {diff.mean().item():.6f}")
print(f"Std diff: {diff.std().item():.6f}")

# Per-token analysis
print("\n=== Per-token analysis (first 10 tokens) ===")
for i in range(min(10, gguf_embeds.shape[1])):
    token_diff = diff[0, i, :].mean().item()
    token_max = diff[0, i, :].max().item()
    gguf_token = gguf_embeds[0, i, :].numpy()
    full_token = full_embeds[0, i, :].numpy()
    token_corr = np.corrcoef(gguf_token, full_token)[0, 1]
    
    print(f"Token {i:3d}: mean_diff={token_diff:.6f}, max_diff={token_max:.6f}, corr={token_corr:.6f}")

# Check if embeddings are just scaled differently
print("\n=== Correlation check ===")
gguf_flat = gguf_embeds.flatten().numpy()
full_flat = full_embeds.flatten().numpy()
correlation = np.corrcoef(gguf_flat, full_flat)[0, 1]
print(f"Correlation: {correlation:.6f}")

# Check scale
scale_ratio = np.std(full_flat) / np.std(gguf_flat)
print(f"Scale ratio (full/gguf): {scale_ratio:.6f}")
