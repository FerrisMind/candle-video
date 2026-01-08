"""
Compare T5 output using ComfyUI-dequantized GGUF weights vs full-precision weights.
This will show what level of error is expected just from Q5_K quantization.
"""
import torch
import numpy as np
from transformers import T5EncoderModel, AutoTokenizer

# Load full-precision T5
model_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder'
model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test prompt
prompt = 'The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.'
inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
input_ids = inputs['input_ids'].to('cuda')
attention_mask = inputs['attention_mask'].to('cuda')

print("Running full-precision T5...")
with torch.no_grad():
    outputs_fp = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    embeddings_fp = outputs_fp.last_hidden_state

# Load Rust GGUF embeddings for comparison
from safetensors import safe_open

print("\nLoading Rust GGUF embeddings...")
with safe_open("t5_gguf_embeddings.safetensors", framework="pt") as f:
    embeddings_rust = f.get_tensor("prompt_embeds")

# Compare
print("\n=== Comparison: Rust GGUF vs Python Full-Precision ===")
embeddings_fp_np = embeddings_fp.cpu().numpy()
embeddings_rust_np = embeddings_rust.numpy()

diff = np.abs(embeddings_fp_np - embeddings_rust_np)
print(f"Max Abs Diff: {diff.max():.6f}")
print(f"Mean Abs Diff: {diff.mean():.6f}")
print(f"Correlation: {np.corrcoef(embeddings_fp_np.flatten(), embeddings_rust_np.flatten())[0,1]:.6f}")

# Find max diff location
max_idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"Max diff at: Batch {max_idx[0]}, Token {max_idx[1]}, Dim {max_idx[2]}")
print(f"  Rust value: {embeddings_rust_np[max_idx]:.6f}")
print(f"  Python value: {embeddings_fp_np[max_idx]:.6f}")

# Per-token analysis for first 10 tokens and token 36
print("\n=== Per-Token Analysis ===")
for t in list(range(10)) + [36]:
    t_rust = embeddings_rust_np[0, t, :]
    t_python = embeddings_fp_np[0, t, :]
    t_diff = np.abs(t_rust - t_python)
    corr = np.corrcoef(t_rust, t_python)[0, 1]
    print(f"Token {t:2d}: Max Diff = {t_diff.max():.4f}, Mean Diff = {t_diff.mean():.6f}, Corr = {corr:.6f}")
