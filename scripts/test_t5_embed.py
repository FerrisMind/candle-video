import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "tp", "diffusers", "src"))
from transformers import T5EncoderModel, AutoTokenizer

model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
text_encoder_path = os.path.join(model_path, "text_encoder")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Loading T5 Encoder on {device} with {dtype}...")
tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
model = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=dtype).to(device)
model.eval()

prompt = "The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon."

print("Encoding prompt...")
inputs = tokenizer(prompt, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings first 10: {embeddings.flatten()[:10].tolist()}")
print(f"Embeddings mean: {embeddings.mean().item():.6f}")
print(f"Embeddings std: {embeddings.std().item():.6f}")

# Save for comparison
np.save("t5_embeddings_python.npy", embeddings.cpu().numpy())
print("Saved to t5_embeddings_python.npy")
