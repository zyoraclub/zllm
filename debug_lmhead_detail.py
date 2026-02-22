#!/usr/bin/env python3
"""Debug lm_head weight loading and final logits computation."""
import numpy as np
import torch
from gguf import GGUFReader
import sys
sys.path.insert(0, '.')
from zllm.engine.quantization import dequantize_tensor
from zllm.engine.inference import ZLLMInferenceEngine
import io, contextlib

print("=== Loading engine ===")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

print(f"lm_head weight shape: {engine.lm_head.weight.shape}")
print(f"lm_head weight stats: mean={engine.lm_head.weight.mean():.6f}, std={engine.lm_head.weight.std():.6f}")

# Load raw output weight from GGUF
print("\n=== Raw GGUF output.weight ===")
reader = GGUFReader('models/tinyllama-1.1b-q4_k_m.gguf')
for t in reader.tensors:
    if t.name == 'output.weight':
        print(f"GGUF shape: {t.shape}")
        print(f"GGUF type: {t.tensor_type}")
        
        raw = dequantize_tensor(bytes(t.data), t.tensor_type.value, tuple(t.shape))
        print(f"Dequantized shape: {raw.shape}")
        print(f"Dequantized stats: mean={raw.mean():.6f}, std={raw.std():.6f}")
        
        # GGUF stores as (hidden, vocab) = (2048, 32000)
        # PyTorch Linear expects (vocab, hidden) = (32000, 2048)
        # So we need to transpose
        raw_transposed = raw.t()
        print(f"Transposed shape: {raw_transposed.shape}")
        
        # Compare with loaded weight
        our_lm_head = engine.lm_head.weight.detach().cpu()
        print(f"\nOur lm_head shape: {our_lm_head.shape}")
        
        diff = (our_lm_head - raw_transposed).abs()
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.6f}")
        print(f"Match? {torch.allclose(our_lm_head, raw_transposed, atol=1e-4)}")
        
        # Check specific token (Paris = 3681 in llama tokenizer typically)
        paris_id = 3681
        print(f"\n=== Token {paris_id} lm_head row ===")
        print(f"Our: {our_lm_head[paris_id, :5]}")
        print(f"Raw transposed: {raw_transposed[paris_id, :5]}")
        break

# Now let's manually compute logits for a hidden state
print("\n=== Manual logits computation test ===")
# Create a simple test hidden state
test_hidden = torch.randn(1, 1, 2048).to(engine.device)

# Compute with our lm_head
our_logits = engine.lm_head(test_hidden)
print(f"Our logits shape: {our_logits.shape}")
print(f"Our logits[0,0,:5]: {our_logits[0,0,:5]}")

# Manually compute with raw weight
raw_weight = raw_transposed.to(engine.device)
manual_logits = torch.nn.functional.linear(test_hidden, raw_weight)
print(f"Manual logits shape: {manual_logits.shape}")
print(f"Manual logits[0,0,:5]: {manual_logits[0,0,:5]}")

logit_diff = (our_logits - manual_logits).abs()
print(f"Logits max diff: {logit_diff.max():.6f}")
