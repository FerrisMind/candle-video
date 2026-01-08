import torch
import numpy as np
from safetensors.torch import load_file

def compare():
    # Load Rust GGUF embeddings
    rust_data = load_file("t5_gguf_embeddings.safetensors")
    # Try multiple possible keys
    if "prompt_embeds" in rust_data:
        rust_emb = rust_data["prompt_embeds"].numpy()
    elif "prompt_embeds_0" in rust_data:
        rust_emb = rust_data["prompt_embeds_0"].numpy()
    else:
        print(f"Keys in rust file: {rust_data.keys()}")
        return
    
    # Load Python full precision embeddings
    python_emb = np.load("t5_embeddings_python.npy")
    
    print(f"Rust shape: {rust_emb.shape}")
    print(f"Python shape: {python_emb.shape}")
    
    # Flatten for global stats
    r_flat = rust_emb.flatten()
    p_flat = python_emb.flatten()
    
    diff = np.abs(r_flat - p_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Find exact location
    max_idx = np.argmax(diff)
    token_idx = max_idx // (rust_emb.shape[1] * rust_emb.shape[2]) # Wait, if it's [B, S, D]
    # Correct indexing for [B, S, D]
    b, s, d = rust_emb.shape
    b_idx = max_idx // (s * d)
    s_idx = (max_idx % (s * d)) // d
    d_idx = max_idx % d
    
    correlation = np.corrcoef(r_flat, p_flat)[0, 1]
    
    print(f"\nMax Abs Diff: {max_diff:.6f}")
    print(f"  At: Batch {b_idx}, Token {s_idx}, Dimension {d_idx}")
    print(f"  Values: Rust {rust_emb[b_idx, s_idx, d_idx]:.6f}, Python {python_emb[b_idx, s_idx, d_idx]:.6f}")
    print(f"Mean Abs Diff: {mean_diff:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    # Per-token analysis for first 10
    print("\nPer-token Correlation (first 10):")
    for i in range(10):
        t_r = rust_emb[0, i, :]
        t_p = python_emb[0, i, :]
        t_corr = np.corrcoef(t_r, t_p)[0, 1]
        t_max = np.max(np.abs(t_r - t_p))
        print(f"Token {i:2}: Corr = {t_corr:.6f}, Max Diff = {t_max:.6f}")

    # Check padding tokens (last token)
    print(f"\nLast token (127) Analysis:")
    t_r = rust_emb[0, 127, :]
    t_p = python_emb[0, 127, :]
    t_corr = np.corrcoef(t_r, t_p)[0, 1]
    t_max = np.max(np.abs(t_r - t_p))
    print(f"Token 127: Corr = {t_corr:.6f}, Max Diff = {t_max:.6f}")


if __name__ == "__main__":
    compare()
