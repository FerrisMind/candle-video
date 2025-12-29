#!/usr/bin/env python3
"""
Simple test: Compare T5 embeddings between our GGUF encoder and diffusers.
"""

import torch
import numpy as np
from pathlib import Path


def test_t5_embeddings():
    """Compare T5 embeddings."""
    from transformers import T5EncoderModel, T5Tokenizer
    
    output_dir = Path("output/debug_t5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompt = "A red apple on a wooden table"
    
    print("Loading T5 from HuggingFace...")
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    
    # Tokenize
    tokens = tokenizer(
        prompt,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    print(f"Token IDs: {tokens['input_ids'][0][:20].tolist()}...")
    
    # Save tokens for comparison
    np.save(str(output_dir / "token_ids.npy"), tokens['input_ids'].numpy())
    print(f"Saved token_ids.npy")
    
    print("\nTo compare with Rust, run inference and check T5 token IDs match.")


if __name__ == "__main__":
    test_t5_embeddings()
