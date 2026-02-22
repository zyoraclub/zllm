#!/usr/bin/env python3
"""Test single full forward pass - no generation loop."""

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

# Clear any existing cache
engine.clear_kv_cache()

prompt = 'The capital of France is'
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], device=engine.device)
print(f"\nPrompt: {prompt}")
print(f"Token IDs: {tokens}")
print(f"Sequence length: {len(tokens)}")

# Single forward pass - no generation
print("\n=== Single Forward Pass ===")
with torch.inference_mode():
    logits = engine.forward(input_ids, use_cache=False)  # Disable cache for clean test

print(f"\nFinal logits shape: {logits.shape}")
print(f"Final logits: mean={logits.float().mean():.6f}, std={logits.float().std():.6f}")
print(f"Logits min={logits.min():.6f}, max={logits.max():.6f}")

# Check for NaN/Inf
nan_count = torch.isnan(logits).sum().item()
inf_count = torch.isinf(logits).sum().item()
print(f"NaN count: {nan_count}, Inf count: {inf_count}")

# Get predicted token
if nan_count == 0 and inf_count == 0:
    next_token_logits = logits[0, -1, :]
    next_token = next_token_logits.argmax().item()
    print(f"\nPredicted next token ID: {next_token}")
    print(f"Predicted next token: '{tokenizer.decode([next_token])}'")
