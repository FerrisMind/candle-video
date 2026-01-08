import torch
from transformers import T5EncoderModel
import numpy as np
from gguf import GGUFReader

# Load Python T5 O weights
model_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder'
model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
python_o_weight = model.encoder.block[0].layer[0].SelfAttention.o.weight.detach().cpu().numpy()
print(f"Python O weight shape: {python_o_weight.shape}")

# Load GGUF and find O weight
gguf_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder_gguf\t5-v1_1-xxl-encoder-Q5_K_M.gguf'
reader = GGUFReader(gguf_path)

for tensor in reader.tensors:
    if 'enc.blk.0.attn_o.weight' in tensor.name:
        print(f"\nGGUF O weight: name={tensor.name}, shape={tensor.shape}, type={tensor.tensor_type}")
        
        # Dequantize the weight (this depends on the GGUF library's capabilities)
        # For now, just compare shapes and patterns
        gguf_data = tensor.data
        print(f"GGUF data shape: {gguf_data.shape}")
        
        # Check if GGUF stores in different order by comparing pattern
        # If GGUF is transposed, we'd see column-major vs row-major pattern
        break

# Focus on what we can compare: compute the O projection manually
print("\n=== Manual O projection test ===")

# Create test input
test_input = torch.randn(1, 1, 4096, dtype=torch.float32)

# Python O projection
python_out = test_input @ python_o_weight.T
print(f"Python O projection output first 5: {python_out.flatten()[:5].tolist()}")

# Let me also check the exact memory layout
print(f"\nPython O weight memory layout check:")
print(f"  [0, :5] = {python_o_weight[0, :5].tolist()}")
print(f"  [:5, 0] = {python_o_weight[:5, 0].tolist()}")
print(f"  [100, 100:105] = {python_o_weight[100, 100:105].tolist()}")

# Check strides to understand memory layout
tensor = model.encoder.block[0].layer[0].SelfAttention.o.weight
print(f"\nPython O weight stride: {tensor.stride()}")
print(f"Expected row-major stride for [4096, 4096]: (4096, 1)")
