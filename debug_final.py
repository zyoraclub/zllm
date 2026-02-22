#!/usr/bin/env python3
"""Final debug - check embedding tying, intermediate size, tokenizer."""

import sys
sys.path.insert(0, '/Users/redfoxhotels/zllm')

import torch
import numpy as np
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf
from llama_cpp import Llama

print("=== Loading Models ===")
parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)
engine = ZLLMInferenceEngine('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Load llama.cpp reference
llm = Llama(model_path='/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf', n_ctx=64, verbose=False)

print("\n=== Architecture Check ===")
print(f"intermediate_size: {engine.intermediate_size}")
print(f"metadata feed_forward_length: {engine.metadata.feed_forward_length}")
print(f"hidden_size: {engine.hidden_size}")
print(f"num_heads: {engine.num_heads}")
print(f"num_kv_heads: {engine.num_kv_heads}")
print(f"head_dim: {engine.hidden_size // engine.num_heads}")

print("\n=== Embedding/LM_Head Weight Tying ===")
emb_weight = engine.embed_tokens.weight.data.float()
lm_head_weight = engine.lm_head.weight.data.float()
print(f"embed_tokens shape: {emb_weight.shape}")
print(f"lm_head shape: {lm_head_weight.shape}")
print(f"Are they exactly equal? {torch.equal(emb_weight, lm_head_weight)}")
print(f"Are they close? {torch.allclose(emb_weight, lm_head_weight, atol=1e-4)}")
if not torch.equal(emb_weight, lm_head_weight):
    diff = (emb_weight - lm_head_weight).abs()
    print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

print("\n=== Norm Weight Check ===")
print(f"Final norm weight: mean={engine.norm.weight.mean():.6f}, std={engine.norm.weight.std():.6f}")
print(f"First 10: {engine.norm.weight[:10].tolist()}")

print("\n=== Tokenizer Comparison ===")
prompt = "The capital of France is"
our_tokens = tokenizer.encode(prompt)
# llama.cpp tokenization
llm_tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=True)
print(f"Our tokens:      {our_tokens}")
print(f"llama.cpp tokens: {llm_tokens}")
print(f"Tokens match: {our_tokens == llm_tokens}")

print("\n=== FFN Weight Shapes ===")
layer0 = engine._load_layer(0)
print(f"gate_proj: {layer0.gate_proj.weight.shape}")
print(f"up_proj:   {layer0.up_proj.weight.shape}")
print(f"down_proj: {layer0.down_proj.weight.shape}")

# Check if gate/up are swapped
print("\n=== Gate/Up Weight Stats (layer 0) ===")
print(f"gate_proj: mean={layer0.gate_proj.weight.mean():.6f}, std={layer0.gate_proj.weight.std():.6f}")
print(f"up_proj:   mean={layer0.up_proj.weight.mean():.6f}, std={layer0.up_proj.weight.std():.6f}")
print(f"down_proj: mean={layer0.down_proj.weight.mean():.6f}, std={layer0.down_proj.weight.std():.6f}")

# Check GGUF tensor names for FFN
print("\n=== GGUF FFN Tensor Names (layer 0) ===")
ffn_names = [k for k in parser.tensors.keys() if 'blk.0' in k and ('ffn' in k or 'mlp' in k)]
for name in sorted(ffn_names):
    info = parser.tensors[name]
    print(f"  {name}: {info.shape}")
