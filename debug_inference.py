"""Debug script to trace inference computation."""

import sys
sys.path.insert(0, '.')
import torch
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf

print("="*60)
print("ZLLM INFERENCE DEBUG")
print("="*60)

# Load tokenizer
parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)

# Test tokenization
prompt = 'Hello'
tokens = tokenizer.encode(prompt)
print(f'\nTokens for "{prompt}": {tokens}')
print(f'Decoded back: {tokenizer.decode(tokens)}')

# Create engine (suppress weight loading messages)
import io
import contextlib

print("\nLoading engine...")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')
print("Engine loaded.")

# Test embedding layer
print("\n" + "="*60)
print("STEP 1: EMBEDDING LOOKUP")
print("="*60)
input_ids = torch.tensor([tokens], device=engine.device)
print(f'Input IDs: {input_ids}')
print(f'Input shape: {input_ids.shape}')

embed = engine.embed_tokens(input_ids)
print(f'Embed shape: {embed.shape}')
print(f'Embed dtype: {embed.dtype}')
print(f'Embed mean: {embed.float().mean():.6f}')
print(f'Embed std: {embed.float().std():.6f}')
print(f'Embed range: [{embed.min():.4f}, {embed.max():.4f}]')
print(f'First token, first 10 dims: {embed[0, 0, :10].tolist()}')

# Test first layer norm
print("\n" + "="*60)
print("STEP 2: FIRST LAYER")
print("="*60)

# Load layer 0
layer = engine._load_layer(0)
print(f'Layer 0 loaded')

# Apply input layernorm
normed = layer.input_layernorm(embed)
print(f'After LayerNorm - mean: {normed.float().mean():.6f}, std: {normed.float().std():.6f}')

# Get Q projection
q = layer.q_proj(normed)
print(f'Q shape: {q.shape}')
print(f'Q mean: {q.float().mean():.6f}, std: {q.float().std():.6f}')

# Full forward through first layer
print("\n" + "="*60)
print("STEP 3: FULL FORWARD (first layer)")
print("="*60)
batch_size, seq_len = input_ids.shape
position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)

hidden = embed
out, kv = layer(
    hidden.to(engine.dtype),
    engine.cos_cache.to(engine.device),
    engine.sin_cache.to(engine.device),
    position_ids,
    attention_mask=None,
    kv_cache=None,
)
print(f'Layer 0 output shape: {out.shape}')
print(f'Layer 0 output mean: {out.float().mean():.6f}, std: {out.float().std():.6f}')
print(f'Layer 0 output range: [{out.min():.4f}, {out.max():.4f}]')

# Check for NaN/Inf
print(f'Has NaN: {torch.isnan(out).any()}')
print(f'Has Inf: {torch.isinf(out).any()}')

# Test full forward
print("\n" + "="*60)
print("STEP 4: FULL MODEL FORWARD")
print("="*60)
engine.clear_kv_cache()
logits = engine.forward(input_ids, use_cache=False)
print(f'Logits shape: {logits.shape}')
print(f'Logits mean: {logits.float().mean():.6f}, std: {logits.float().std():.6f}')
print(f'Logits range: [{logits.min():.4f}, {logits.max():.4f}]')

# Get top predictions
probs = torch.softmax(logits[0, -1].float(), dim=-1)
top_k = torch.topk(probs, k=10)
print(f'\nTop 10 predictions for next token:')
for i, (prob, idx) in enumerate(zip(top_k.values.tolist(), top_k.indices.tolist())):
    token = tokenizer.vocab[idx] if idx < len(tokenizer.vocab) else f'[{idx}]'
    print(f'  {i+1}. "{token}" (id={idx}): {prob*100:.2f}%')
