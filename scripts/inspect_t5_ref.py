import torch
from transformers import T5EncoderModel, T5Tokenizer
import safetensors.torch
import sys
import os

def main():
    model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
    prompt = "A red cube"
    
    tokenizer_path = os.path.join(model_path, "tokenizer")
    text_encoder_path = os.path.join(model_path, "text_encoder")
    
    print(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    except:
        print(f"Failed, trying {text_encoder_path}")
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_path)

    print(f"Loading text encoder from {text_encoder_path}")
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" # Force CPU to match potential quantization discrepancies? No, let's use what's available.
    text_encoder.to(device)

    print(f"Encoding prompt: '{prompt}'")
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    
    print(f"Token IDs (first 20): {text_inputs.input_ids[0, :20].tolist()}")
    
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]

    prompt_embeds = prompt_embeds.cpu()
    print(f"Embeddings shape: {prompt_embeds.shape}")
    print(f"First token embedding mean: {prompt_embeds[0, 0].mean()}")
    print(f"First token embedding std: {prompt_embeds[0, 0].std()}")

    save_path = "t5_embeddings_py.safetensors"
    print(f"Saving embeddings to {save_path}")
    safetensors.torch.save_file({"prompt_embeds": prompt_embeds}, save_path)

if __name__ == "__main__":
    main()
