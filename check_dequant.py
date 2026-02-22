#!/usr/bin/env python3
"""Check dequantization weight stats."""
import sys
import importlib.util

# Load modules directly to avoid heavy __init__.py
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

import numpy as np
import torch

gguf_parser = load_mod('gguf_parser', '/Users/redfoxhotels/zllm/zllm/engine/gguf_parser.py')
quantization = load_mod('quantization', '/Users/redfoxhotels/zllm/zllm/engine/quantization.py')
tokenizer_mod = load_mod('tokenizer', '/Users/redfoxhotels/zllm/zllm/engine/tokenizer.py')
inference = load_mod('inference', '/Users/redfoxhotels/zllm/zllm/engine/inference.py')

print('=== DEQUANT WEIGHT STATS ===')

parser = gguf_parser.GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Q weight
info = parser.tensors['blk.0.attn_q.weight']
raw = parser.load_tensor_raw('blk.0.attn_q.weight')
w = quantization.dequantize_tensor(raw, info.dtype.value, info.shape)
print(f'blk.0.attn_q.weight:')
print(f'  Shape: {w.shape}, Type: {info.dtype}')
print(f'  Mean: {w.float().mean():.6f}, Std: {w.float().std():.6f}')
print(f'  Min: {w.float().min():.6f}, Max: {w.float().max():.6f}')

# Test inference
print()
print('=== INFERENCE TEST ===')
tokenizer = tokenizer_mod.load_tokenizer_from_gguf(parser)
engine = inference.ZLLMInferenceEngine('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

tokens = tokenizer.encode('The capital of France is')
print(f'Input tokens: {tokens}')
output = engine.generate(tokens, max_new_tokens=10, temperature=0.1)
print(f'Output: {tokenizer.decode(output)}')
