"""
Compare our T5 GGUF implementation with llama-cpp-python.
ComfyUI uses llama-cpp-python for T5, so if results differ, we have a bug.
"""
import torch
import numpy as np

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    print("llama-cpp-python available!")
except ImportError:
    print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    print("Alternatively, we can compare with ggml-python or manual dequantization.")
    exit(1)

# The GGUF file
gguf_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder_gguf\t5-v1_1-xxl-encoder-Q5_K_M.gguf"

print(f"Loading T5 GGUF from {gguf_path}...")

# Note: llama-cpp-python is designed for decoder models (LLMs)
# T5 encoder-only GGUF might need special handling
# Let's check if we can at least read the model
try:
    model = Llama(model_path=gguf_path, embedding=True, n_ctx=512, verbose=True)
    print(f"Model loaded successfully!")
except Exception as e:
    print(f"Failed to load with llama-cpp: {e}")
    print("\nllama-cpp-python doesn't support T5 encoder models directly.")
    print("ComfyUI likely uses a custom T5 GGUF loader, not standard llama-cpp.")
