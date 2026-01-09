#!/usr/bin/env python3
"""
Generate T5 prompt encoding reference from original transformers T5.
"""

import torch
from transformers import T5EncoderModel, T5Tokenizer
from safetensors.torch import save_file

def gen_t5_ref():
    print("Loading T5 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use same model that Lightricks uses
    model_id = "google/t5-v1_1-xxl"
    
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    
    print(f"Loading model from {model_id}...")
    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    # Test prompt (same as video generation)
    prompt = "A man walks towards a window, looks out, and then turns around."
    
    print(f"Encoding prompt: {prompt[:50]}...")
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding="max_length",
        max_length=256,
        truncation=True,
    ).to(device)
    
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"First 20 input IDs: {inputs.input_ids[0, :20].tolist()}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings mean: {embeddings.mean().item():.6f}")
    print(f"Embeddings std: {embeddings.std().item():.6f}")
    print(f"Embeddings range: {embeddings.min().item():.4f} to {embeddings.max().item():.4f}")
    
    # Save
    tensors = {
        "prompt_text": prompt,
        "input_ids": inputs.input_ids.cpu(),
        "embeddings": embeddings.cpu().float(),
    }
    
    save_file(tensors, "gen_t5_ref.safetensors")
    print("\nDone. Saved to gen_t5_ref.safetensors")

if __name__ == "__main__":
    gen_t5_ref()
