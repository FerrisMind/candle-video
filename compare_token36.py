import torch
from transformers import T5EncoderModel, AutoTokenizer

# Load model and tokenizer from correct paths
model_path = r'c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5\text_encoder'
model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = 'The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.'
inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
input_ids = inputs['input_ids'].to('cuda')
attention_mask = inputs['attention_mask'].to('cuda')

print(f"Input IDs first 20: {input_ids[0, :20].tolist()}")

with torch.no_grad():
    block0 = model.encoder.block[0]
    hidden = model.encoder.embed_tokens(input_ids)
    normed = block0.layer[0].layer_norm(hidden)
    
    # Compute attention manually
    attn = block0.layer[0].SelfAttention
    q = attn.q(normed)
    k = attn.k(normed)
    v = attn.v(normed)
    
    # Reshape: [1, 128, 4096] -> [1, 64, 128, 64]
    q = q.view(1, 128, 64, 64).transpose(1, 2)
    k = k.view(1, 128, 64, 64).transpose(1, 2)
    v = v.view(1, 128, 64, 64).transpose(1, 2)
    
    pos_bias = attn.compute_bias(128, 128)
    extended_mask = model.get_extended_attention_mask(attention_mask, input_ids.shape, 'cuda')
    
    scores = torch.matmul(q, k.transpose(3, 2))
    scores = scores + pos_bias
    scores = scores + extended_mask
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_output = torch.matmul(attn_weights, v)
    
    # Reshape back: [1, 64, 128, 64] -> [1, 128, 4096]
    attn_output = attn_output.transpose(1, 2).reshape(1, 128, -1)
    
    print(f"\nPython Token 0 matmul V first 5: {attn_output[0, 0, :5].tolist()}")
    print(f"Python Token 36 matmul V first 5: {attn_output[0, 36, :5].tolist()}")
    
    # Compare with Rust values
    rust_t36 = [0.06057674, -0.026639095, 0.013187891, 0.066518895, -0.07847871]
    python_t36 = attn_output[0, 36, :5].tolist()
    
    print(f"\nRust   Token 36 matmul V: {rust_t36}")
    print(f"Python Token 36 matmul V: {python_t36}")
    print(f"Diffs: {[round(r - p, 6) for r, p in zip(rust_t36, python_t36)]}")
