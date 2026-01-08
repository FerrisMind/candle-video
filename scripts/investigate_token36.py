"""
Investigate Token 36 anomaly - what makes it special?
"""
import torch
from transformers import T5EncoderModel, AutoTokenizer
import numpy as np

# Load tokenizer and model
model_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder'
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = 'The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.'
inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
input_ids = inputs['input_ids'][0]
attention_mask = inputs['attention_mask'][0]

print("=== Token Analysis ===")
print(f"Total tokens: {input_ids.shape[0]}")
print(f"Non-padding tokens: {attention_mask.sum().item()}")

# Find where padding starts
padding_start = (attention_mask == 0).nonzero()
if len(padding_start) > 0:
    padding_start_idx = padding_start[0].item()
    print(f"Padding starts at token: {padding_start_idx}")
else:
    padding_start_idx = 128
    print("No padding in this sequence")

# Decode tokens around 36
print(f"\n=== Tokens around position 36 ===")
for i in range(max(0, 33), min(128, 40)):
    token_id = input_ids[i].item()
    token_str = tokenizer.decode([token_id])
    mask = attention_mask[i].item()
    special = ""
    if i == 36:
        special = " <-- TOKEN 36"
    if i == padding_start_idx:
        special = " <-- PADDING STARTS"
    print(f"  Position {i:3d}: ID={token_id:5d}, mask={mask}, text='{token_str}'{special}")

# Check dimension 1478
print(f"\n=== Dimension 1478 Analysis ===")
print(f"Dimension 1478 out of 4096 total dimensions (d_model)")
print(f"This corresponds to head {1478 // 64} (64 heads, 64 dim each), position {1478 % 64} within head")

# Load embeddings and analyze Token 36 vs others
from safetensors import safe_open

with safe_open("t5_gguf_embeddings.safetensors", framework="pt") as f:
    rust_embeds = f.get_tensor("prompt_embeds")

model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32).to('cuda')
with torch.no_grad():
    python_embeds = model.encoder(
        input_ids=inputs['input_ids'].to('cuda'),
        attention_mask=inputs['attention_mask'].to('cuda')
    ).last_hidden_state.cpu()

# Compare dimension 1478 across all tokens
print(f"\n=== Dimension 1478 across all tokens ===")
dim = 1478
rust_dim = rust_embeds[0, :, dim].numpy()
python_dim = python_embeds[0, :, dim].numpy()
diff_dim = np.abs(rust_dim - python_dim)

print(f"{'Token':>5} {'Rust':>10} {'Python':>10} {'Diff':>10} {'Note':>20}")
for i in range(min(50, len(rust_dim))):
    note = ""
    if i == 36:
        note = "<-- PROBLEM"
    if diff_dim[i] > 0.1:
        note = f"<-- LARGE DIFF"
    print(f"{i:5d} {rust_dim[i]:10.4f} {python_dim[i]:10.4f} {diff_dim[i]:10.4f} {note:>20}")

# Check if Token 36 has issues across ALL dimensions
print(f"\n=== Token 36 dimension-wise analysis ===")
t36_rust = rust_embeds[0, 36, :].numpy()
t36_python = python_embeds[0, 36, :].numpy()
t36_diff = np.abs(t36_rust - t36_python)

# Find top 10 worst dimensions for Token 36
worst_dims = np.argsort(t36_diff)[-10:][::-1]
print(f"Top 10 worst dimensions for Token 36:")
for dim in worst_dims:
    print(f"  Dim {dim:4d}: Rust={t36_rust[dim]:8.4f}, Python={t36_python[dim]:8.4f}, Diff={t36_diff[dim]:8.4f}")
