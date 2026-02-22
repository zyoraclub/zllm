#!/usr/bin/env python3
"""Debug all layers - find where divergence starts."""

import sys
sys.path.insert(0, '/Users/redfoxhotels/zllm')

import torch
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf

print("=== Loading Model ===")
parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)
engine = ZLLMInferenceEngine('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Check critical metadata
print(f"\n=== Critical Metadata ===")
print(f"rope_dimension_count: {engine.metadata.rope_dimension_count}")
print(f"head_dim: {engine.hidden_size // engine.num_heads}")
print(f"rope_freq_base: {engine.metadata.rope_freq_base}")

# Prepare input
prompt = 'The capital of France is'
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], device=engine.device)
seq_len = input_ids.shape[1]
position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)

print(f"\nPrompt: {prompt}")
print(f"Tokens: {tokens}")

# Manual forward with per-layer stats
print(f"\n=== Per-Layer Stats (use_cache=False) ===")

# Build mask
mask = torch.full((seq_len, seq_len), float("-inf"), device=engine.device)
mask = torch.triu(mask, diagonal=1)
mask = mask.unsqueeze(0).unsqueeze(0)

# Embedding
hidden_states = engine.embed_tokens(input_ids)
print(f"Embed:    mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

# Process each layer
for layer_idx in range(engine.num_layers):
    layer = engine._load_layer(layer_idx)
    hidden_states, _ = layer(
        hidden_states,
        engine.cos_cache,
        engine.sin_cache,
        position_ids,
        attention_mask=mask,
        kv_cache=None,  # No cache
    )
    if layer_idx < 3 or layer_idx >= engine.num_layers - 3:
        print(f"Layer {layer_idx:2d}: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")
    elif layer_idx == 3:
        print("...")

# Final norm and lm_head
hidden_states = engine.norm(hidden_states)
print(f"Norm:     mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

logits = engine.lm_head(hidden_states)
print(f"Logits:   mean={logits.mean():.6f}, std={logits.std():.6f}")

# Check predictions
print(f"\n=== Prediction Analysis ===")
last_logits = logits[0, -1, :].float()
top5 = torch.topk(last_logits, 5)
print("Top-5 predictions:")
for i, (logit, idx) in enumerate(zip(top5.values, top5.indices)):
    token = tokenizer.decode([idx.item()])
    print(f"  {i+1}. '{token}' (id={idx.item()}, logit={logit.item():.4f})")

# Paris check
paris_id = tokenizer.encode("Paris")[-1]
print(f"\nParis (id={paris_id}) logit: {last_logits[paris_id].item():.4f}")
paris_rank = (last_logits > last_logits[paris_id]).sum().item() + 1
print(f"Paris rank: {paris_rank} / {logits.shape[-1]}")
