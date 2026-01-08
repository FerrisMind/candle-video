"""
Layer-by-layer comparison of T5 GGUF (Rust) vs Full T5 (Python).
Run this after running dump_t5_embeddings to identify where outputs diverge.
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "tp", "diffusers", "src"))
from transformers import T5EncoderModel, AutoTokenizer

model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
text_encoder_path = os.path.join(model_path, "text_encoder")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Python T5
print(f"Loading Python T5...")
tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
model = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.float32).to(device)
model.eval()

print(f"\n=== T5 Config ===")
print(f"Epsilon: {model.config.layer_norm_epsilon}")
print(f"Dense act fn: {model.config.dense_act_fn}")
print(f"Feed forward proj: {model.config.feed_forward_proj}")
print(f"Num layers: {model.config.num_layers}")

prompt = "The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon."

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs first 20: {input_ids[0, :20].tolist()}")

# Get token embeddings (before any transformer layers)
with torch.no_grad():
    # Access embedding layer directly
    embeddings = model.encoder.embed_tokens(input_ids)
    
print(f"\n=== Token Embeddings (before transformer layers) ===")
print(f"Shape: {embeddings.shape}")
print(f"First 10 values: {embeddings.flatten()[:10].tolist()}")
print(f"Mean: {embeddings.mean().item():.6f}")

# Prepare extended attention mask
# T5 uses additive bias mask: 0 for keep, -1e9 for mask
extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ids.shape, device)

# Get output after block 0
with torch.no_grad():
    hidden = embeddings
    position_bias = None
    
    # Run block 0 only
    block0 = model.encoder.block[0]
    
    # Pre-norm
    normed = block0.layer[0].layer_norm(hidden)
    print(f"\n=== After block 0 layer_norm (pre-attention) ===")
    print(f"First 10: {normed.flatten()[:10].tolist()}")
    
    # SelfAttention
    rel_bias_obj = block0.layer[0].SelfAttention
    
    # Get position bias
    pos_bias = rel_bias_obj.compute_bias(128, 128)
    print(f"\n=== Block 0 Position Bias (first head, top 5x5) ===")
    print(pos_bias[0, 0, :5, :5])
    
    # Manually compute scores
    q = rel_bias_obj.q(normed)
    k = rel_bias_obj.k(normed)
    
    batch_size, seq_length, _ = hidden.shape
    q = q.view(batch_size, seq_length, 64, 64).transpose(1, 2)
    k = k.view(batch_size, seq_length, 64, 64).transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(3, 2))
    scores += pos_bias
    
    # Add attention mask
    if extended_attention_mask is not None:
        scores += extended_attention_mask
        
    print(f"\n=== Block 0 Scores before softmax (first head, top 5x5) ===")
    print(scores[0, 0, :5, :5])
    
    # Attention output
    attn_output_layer = block0.layer[0](hidden, attention_mask=extended_attention_mask)[0]
    print(f"\n=== After block 0 attention (with residual) ===")
    print(f"First 10: {attn_output_layer.flatten()[:10].tolist()}")
    
    # Introspect Attention projection outputs
    v_out = rel_bias_obj.v(normed)
    print(f"=== Block 0 Attention V output first 10 ===")
    print(f"{v_out.flatten()[:10].tolist()}")

    # After matmul V but before O
    # attn_weights = [1, 64, 128, 128], v_states = [1, 64, 128, 64]
    # Re-extract from forward to be sure of logic
    query_states = rel_bias_obj.q(normed)
    key_states = rel_bias_obj.k(normed)
    value_states = rel_bias_obj.v(normed)
    query_states = query_states.view(1, 128, 64, 64).transpose(1, 2)
    key_states = key_states.view(1, 128, 64, 64).transpose(1, 2)
    value_states = value_states.view(1, 128, 64, 64).transpose(1, 2)
    scores = torch.matmul(query_states, key_states.transpose(3, 2))
    scores += pos_bias
    if extended_attention_mask is not None:
        scores += extended_attention_mask
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_output_pre_o = torch.matmul(attn_weights, value_states)
    attn_output_pre_o = attn_output_pre_o.transpose(1, 2).reshape(1, 128, -1)
    print(f"=== Block 0 Attn matmul V first 10 ===")
    print(f"{attn_output_pre_o.flatten()[:10].tolist()}")
    t36_start = 36 * 4096
    print(f"=== Block 0 Token 36 matmul V first 5 ===")
    print(f"{attn_output_pre_o.flatten()[t36_start:t36_start+5].tolist()}")


    # Feed-forward intermediates
    normed_ffn = block0.layer[1].layer_norm(attn_output_layer)
    ffn_module = block0.layer[1].DenseReluDense
    gate_out_py = ffn_module.wi_0(normed_ffn)
    print(f"\n=== Block 0 FFN Gate output (before act) ===")
    print(f"First 10: {gate_out_py.flatten()[:10].tolist()}")

    # Full block 0
    layer_outputs = block0(hidden, attention_mask=extended_attention_mask, position_bias=position_bias)
    after_block0 = layer_outputs[0]
    
    print(f"\n=== After block 0 (full) ===")
    print(f"First 10: {after_block0.flatten()[:10].tolist()}")
    print(f"Mean: {after_block0.mean().item():.6f}")



with torch.no_grad():
    # Capture intermediate features for Token 36
    hidden = model.encoder.embed_tokens(input_ids)
    
    # Block 0
    block0 = model.encoder.block[0]
    # In HF, relative_attention_bias is inside SelfAttention. Use compute_bias to handle bucket logic.
    pos_bias = block0.layer[0].SelfAttention.compute_bias(input_ids.shape[1], input_ids.shape[1])

    
    # Introspect Token 36 in Block 0
    t36_idx = 36
    print(f"\n=== Block 0 Token {t36_idx} Intermediates ===")
    print(f"  Normed first 5: {normed[0, t36_idx, :5].tolist()}")
    
    # Pre-O output
    # self.SelfAttention returns (attn_output, pos_bias) if we don't pass pos_bias
    # but block0.layer[0] passes it.
    # Let's extract attn_output before resid
    attn_layer = block0.layer[0]
    normed_hidden = attn_layer.layer_norm(hidden)
    attn_output_pre_resid = attn_layer.SelfAttention(normed_hidden, mask=extended_attention_mask, position_bias=pos_bias)[0]
    print(f"  O output (before residual) first 5: {attn_output_pre_resid[0, t36_idx, :5].tolist()}")
    
    attn_out_full = attn_output_pre_resid + hidden
    print(f"  After Attention (hidden + O) first 5: {attn_out_full[0, t36_idx, :5].tolist()}")

    
    ffn_normed = block0.layer[1].layer_norm(attn_out_full)
    gate_out = block0.layer[1].DenseReluDense.wi_0(ffn_normed)
    print(f"  FFN Gate first 5: {gate_out[0, t36_idx, :5].tolist()}")
    
    # Full encoder pass
    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    final_embeddings = outputs.last_hidden_state
    
    # Block 23 (last block) intermediates
    # We can't easily extract Block 23 intermediates from a black-box forward
    # but we can check final state for token 36
    print(f"\n=== Final Encoder Output Token 36 ===")
    print(f"  First 10: {final_embeddings[0, t36_idx, :10].tolist()}")
    print(f"  Pos 1478: {final_embeddings[0, t36_idx, 1478].item():.6f}")
    
print(f"\n=== Final Embeddings (output of encoder) ===")
print(f"Shape: {final_embeddings.shape}")
print(f"Token 0 First 10: {final_embeddings[0, 0, :10].tolist()}")
print(f"Token 50 First 10: {final_embeddings[0, 50, :10].tolist()}")
print(f"Mean: {final_embeddings.mean().item():.6f}")

np.save("t5_embeddings_python.npy", final_embeddings.cpu().numpy().astype(np.float32))
print("Saved to t5_embeddings_python.npy")

# Also print embedding weight stats
emb_weight = model.encoder.embed_tokens.weight
print(f"\n=== Embedding Weight ===")
print(f"Shape: {emb_weight.shape}")
print(f"First 10 values: {emb_weight.flatten()[:10].tolist()}")
print(f"Mean: {emb_weight.mean().item():.6f}")
print(f"Std: {emb_weight.std().item():.6f}")
