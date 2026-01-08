import torch
import safetensors.torch
import numpy as np

def main():
    try:
        rust_data = safetensors.torch.load_file("t5_embeddings_rust.safetensors")
        py_data = safetensors.torch.load_file("t5_embeddings_py.safetensors")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    rust_emb = rust_data["prompt_embeds"].float()
    py_emb = py_data["prompt_embeds"].float()

    print(f"Rust Shape: {rust_emb.shape}")
    print(f"Py Shape:   {py_emb.shape}")

    # Ensure shapes match
    if rust_emb.shape != py_emb.shape:
        print("Shapes do not match!")
        return

    diff = (rust_emb - py_emb).abs()
    mad = diff.mean().item()
    max_diff = diff.max().item()

    print(f"Mean Absolute Difference (MAD): {mad:.6f}")
    print(f"Max Difference: {max_diff:.6f}")

    # Cosine Similarity
    rust_flat = rust_emb.view(1, -1)
    py_flat = py_emb.view(1, -1)
    cos_sim = torch.nn.functional.cosine_similarity(rust_flat, py_flat).item()
    print(f"Global Cosine Similarity: {cos_sim:.6f}")

    # Per-token cosine similarity
    token_cos_sim = torch.nn.functional.cosine_similarity(rust_emb[0], py_emb[0], dim=1)
    print(f"Avg Per-Token Cosine Similarity: {token_cos_sim.mean().item():.6f}")
    print(f"Min Per-Token Cosine Similarity: {token_cos_sim.min().item():.6f}")

    # Ratio of norms
    rust_norm = torch.norm(rust_emb)
    py_norm = torch.norm(py_emb)
    print(f"Norm Ratio (Rust/Py): {rust_norm/py_norm:.6f}")

if __name__ == "__main__":
    main()
