#!/usr/bin/env python3
"""Debug native engine - compare dequant against reference."""
import sys
# Import directly to avoid heavy transformers loading
sys.path.insert(0, '/Users/redfoxhotels/zllm')

import torch
import numpy as np

# Direct imports
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.quantization import dequantize_tensor

print('=== DEQUANTIZATION COMPARISON ===')

# Our dequant
parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')
info = parser.tensors['blk.0.attn_q.weight']
raw = parser.load_tensor_raw('blk.0.attn_q.weight')
ours = dequantize_tensor(raw, info.dtype.value, info.shape)

print(f'Tensor: blk.0.attn_q.weight')
print(f'GGUF Shape: {info.shape}')
print(f'GGUF Type: {info.dtype}')
print()
print('OUR DEQUANT:')
print(f'  Shape: {ours.shape}')
print(f'  Mean: {ours.float().mean():.6f}')
print(f'  Std: {ours.float().std():.6f}')
print(f'  Min: {ours.float().min():.6f}')
print(f'  Max: {ours.float().max():.6f}')
print(f'  First 10: {ours.flatten()[:10].tolist()}')

# Check multiple layers
print()
print('=== WEIGHT STD ACROSS LAYERS ===')
for layer_idx in [0, 5, 10, 15, 20]:
    name = f'blk.{layer_idx}.attn_q.weight'
    if name in parser.tensors:
        info = parser.tensors[name]
        raw = parser.load_tensor_raw(name)
        w = dequantize_tensor(raw, info.dtype.value, info.shape)
        print(f'Layer {layer_idx} Q: mean={w.float().mean():.6f}, std={w.float().std():.6f}')

# Check embedding
print()
print('=== EMBEDDING ===')
info = parser.tensors['token_embd.weight']
raw = parser.load_tensor_raw('token_embd.weight')
emb = dequantize_tensor(raw, info.dtype.value, info.shape)
print(f'Shape: {emb.shape}')
print(f'Mean: {emb.float().mean():.6f}')
print(f'Std: {emb.float().std():.6f}')

# Check output weight
print()
print('=== OUTPUT WEIGHT ===')
info = parser.tensors['output.weight']
raw = parser.load_tensor_raw('output.weight')
out = dequantize_tensor(raw, info.dtype.value, info.shape)
print(f'Shape: {out.shape}')
print(f'Mean: {out.float().mean():.6f}')
print(f'Std: {out.float().std():.6f}')
