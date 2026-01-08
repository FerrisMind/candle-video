"""
Debug T5 GGUF weights - check if they match Python T5 weights.
"""
import torch
import numpy as np
from safetensors import safe_open
import sys
import os
from gguf import GGUFReader

sys.path.insert(0, os.path.join(os.getcwd(), "tp", "diffusers", "src"))
from transformers import T5EncoderModel

model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
text_encoder_path = os.path.join(model_path, "text_encoder")
gguf_path = os.path.join(model_path, "text_encoder_gguf", "t5-v1_1-xxl-encoder-Q5_K_M.gguf")

# Load GGUF
print(f"Loading GGUF from {gguf_path}...")
reader = GGUFReader(gguf_path)

# Print tensor names and shapes
print("\nGGUF tensors:")
for tensor in reader.tensors[:20]:
    print(f"  {tensor.name}: shape={tensor.shape}")

# Load Python T5 for comparison
print("\nLoading Python T5...")
model = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.float32)
block = model.encoder.block[0]


# Python T5 relative bias
print("\nPython T5 relative bias weights (Bucket x Head):")
# In HF Transformers, relative_attention_bias is in the first block's SelfAttention layer
rel_bias_obj = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
rel_bias_weights = rel_bias_obj.weight
print(f"  Shape: {rel_bias_weights.shape}")
for b_idx in [0, 1, 16, 17]:
    print(f"  Bucket {b_idx:2} Head 0: {rel_bias_weights[b_idx, 0].item():.6f}")

print("\nPython T5 block 0 Attention weights:")
attn = model.encoder.block[0].layer[0].SelfAttention
print(f"  q.weight: {attn.q.weight.shape}")
print(f"  k.weight: {attn.k.weight.shape}")
print(f"  v.weight: {attn.v.weight.shape}")
print(f"  o.weight: {attn.o.weight.shape}")
print(f"  o[0, :5]: {attn.o.weight[0, :5].tolist()}")
print(f"  o[:5, 0]: {attn.o.weight[:5, 0].tolist()}")
print(f"  o[100, :5]: {attn.o.weight[100, :5].tolist()}")


print("\nPython T5 block 0 FFN weights:")
ffn = model.encoder.block[0].layer[1].DenseReluDense
print(f"  wi_0.weight (gate): {ffn.wi_0.weight.shape}")
print(f"  wi_1.weight (up): {ffn.wi_1.weight.shape}")
print(f"  wo.weight (down): {ffn.wo.weight.shape}")
# Compare FFN values to check for gate/up swaps
print("\nComparing Block 0 FFN weight values:")
print(f"  wi_0[0, :5]: {ffn.wi_0.weight[0, :5].tolist()}")
print(f"  wi_1[0, :5]: {ffn.wi_1.weight[0, :5].tolist()}")



