"""
Test ComfyUI's GGUF dequantization vs my Rust implementation
"""
import torch
import numpy as np
from gguf import GGUFReader

# ComfyUI's Q5_K dequantization (from city96/ComfyUI-GGUF/dequant.py)
QK_K = 256
K_SCALE_SIZE = 12

def split_block_dims(blocks, *args):
    n_blocks = blocks.shape[0]
    dims = list(args) + [blocks.shape[1] - sum(args)]
    return torch.split(blocks, dims, dim=-1)

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q5_K(blocks, dtype=torch.float32):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))

# Load GGUF
gguf_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder_gguf\t5-v1_1-xxl-encoder-Q5_K_M.gguf'
reader = GGUFReader(gguf_path)

# Find O weight tensor
o_weight_tensor = None
for tensor in reader.tensors:
    if tensor.name == 'enc.blk.0.attn_o.weight':
        o_weight_tensor = tensor
        break

if o_weight_tensor is None:
    print("O weight tensor not found!")
    exit(1)

print(f"Found O weight: shape={o_weight_tensor.shape}, type={o_weight_tensor.tensor_type}")

# Get raw data as torch tensor
raw_data = torch.from_numpy(o_weight_tensor.data.copy())
print(f"Raw data shape: {raw_data.shape}")

# Dequantize using ComfyUI method
# The tensor shape from GGUF is [rows, compressed_cols]
# For Q5_K, each block of 256 elements is compressed
n_rows = o_weight_tensor.shape[0]  # 4096
n_cols = 4096  # Original uncompressed columns
blocks_per_row = n_cols // QK_K  # 16 blocks per row

# Reshape to blocks
# Each block has: 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs) = 176 bytes
block_size = 2 + 2 + K_SCALE_SIZE + QK_K // 8 + QK_K // 2
print(f"Expected block size: {block_size}, blocks per row: {blocks_per_row}")

# The data shape is [4096, 2816] = [4096, 16 * 176]
expected_cols = blocks_per_row * block_size
print(f"Expected compressed cols: {expected_cols}, actual: {raw_data.shape[1]}")

if raw_data.shape[1] == expected_cols:
    # Reshape to (n_rows * blocks_per_row, block_size)
    blocks = raw_data.reshape(-1, block_size)
    print(f"Blocks shape: {blocks.shape}")
    
    # Dequantize
    dequantized = dequantize_blocks_Q5_K(blocks)
    print(f"Dequantized shape: {dequantized.shape}")
    
    # Reshape to original shape
    final_weights = dequantized.reshape(n_rows, n_cols)
    print(f"Final weights shape: {final_weights.shape}")
    
    # Compare with Python full-precision weights
    from transformers import T5EncoderModel
    model = T5EncoderModel.from_pretrained(
        r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder',
        torch_dtype=torch.float32
    )
    python_o = model.encoder.block[0].layer[0].SelfAttention.o.weight.detach().cpu().numpy()
    
    # Compare
    comfy_dequant = final_weights.numpy()
    
    print("\n=== ComfyUI Dequantized O[0, :5] ===")
    print(comfy_dequant[0, :5])
    print("\n=== Python Full-Precision O[0, :5] ===")
    print(python_o[0, :5])
    
    diff = np.abs(comfy_dequant - python_o)
    print(f"\n=== Comparison ===")
    print(f"Max Abs Diff: {diff.max():.6f}")
    print(f"Mean Abs Diff: {diff.mean():.6f}")
    print(f"Correlation: {np.corrcoef(comfy_dequant.flatten(), python_o.flatten())[0,1]:.6f}")
else:
    print("Unexpected data layout!")
    print(f"Raw data range: min={raw_data.min()}, max={raw_data.max()}")
