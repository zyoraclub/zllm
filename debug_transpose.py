#!/usr/bin/env python3
"""Test if transpose is causing the issue by comparing with/without."""
import torch
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, '.')

from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.quantization import dequantize_tensor
from zllm.engine.inference import ZLLMInferenceEngine, apply_rope, precompute_rope_cache
from zllm.engine.tokenizer import load_tokenizer_from_gguf
import io, contextlib

print("=== Loading engine ===")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)

# Test input
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
batch_size, seq_len = input_ids.shape

# Get embeddings
hidden = engine.embed_tokens(input_ids)

# Load layer 0 weights directly (without engine's transpose)
from gguf import GGUFReader
reader = GGUFReader('models/tinyllama-1.1b-q4_k_m.gguf')

def load_weight_raw(name):
    for t in reader.tensors:
        if t.name == name:
            w = dequantize_tensor(bytes(t.data), t.tensor_type.value, tuple(t.shape))
            return w.to(engine.device)
    raise KeyError(name)

print("\n=== Weight Loading Test ===")

# Load Q weight
q_weight_raw = load_weight_raw('blk.0.attn_q.weight')
print(f"Q weight raw shape: {q_weight_raw.shape}")

# Engine's loaded Q weight
layer = engine._load_layer(0)
q_weight_engine = layer.q_proj.weight.data
print(f"Engine Q weight shape: {q_weight_engine.shape}")

# Check if transpose is correct
print(f"\nAre they transposed? raw.T == engine: {torch.allclose(q_weight_raw.t().float(), q_weight_engine.float(), atol=1e-3)}")
print(f"Are they equal? raw == engine: {torch.allclose(q_weight_raw.float(), q_weight_engine.float(), atol=1e-3)}")

# Apply normalization
normed = layer.input_layernorm(hidden)
normed = normed.to(q_weight_engine.dtype)

print("\n=== Projection Test ===")

# Test 1: using engine's (transposed) weight
q_transposed = F.linear(normed, q_weight_engine)
print(f"Q with transposed weight: shape={q_transposed.shape}, std={q_transposed.std():.6f}")

# Test 2: using raw (non-transposed) weight - need to do manual matmul
q_raw = torch.matmul(normed, q_weight_raw)
print(f"Q with raw weight: shape={q_raw.shape}, std={q_raw.std():.6f}")

# Which produces better attention?
print("\n=== Attention Quality Test ===")

def test_attention(q_proj, name):
    # Load K
    k_weight_raw = load_weight_raw('blk.0.attn_k.weight')
    k_transposed = F.linear(normed, k_weight_raw.t())  # use transposed
    
    # Reshape
    head_dim = 64
    num_heads = 32
    num_kv_heads = 4
    
    q = q_proj.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k_transposed.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # RoPE
    cos, sin = precompute_rope_cache(head_dim, 2048, 10000.0, engine.device)
    position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
    q, k = apply_rope(q, k, cos, sin, position_ids)
    
    # GQA
    k_rep = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    
    # Attention scores
    scale = 1.0 / math.sqrt(head_dim)
    attn = torch.matmul(q, k_rep.transpose(-2, -1)) * scale
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=engine.device), diagonal=1).bool()
    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    probs = F.softmax(attn, dim=-1)
    
    # Check uniformity
    last_pos_probs = probs[0, 0, -1, :].detach().cpu().numpy()
    entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean().item()
    
    print(f"\n{name}:")
    print(f"  Last position attention: {last_pos_probs}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Pre-softmax range: [{attn[attn > -1e9].min():.4f}, {attn.max():.4f}]")

test_attention(q_transposed, "With transpose (current)")
test_attention(q_raw, "Without transpose (raw)")

# Also test what llama.cpp-style attention looks like
print("\n=== Comparing Q weights directly ===")
print(f"Engine Q weight [0, :5]: {q_weight_engine[0, :5]}")
print(f"Raw Q weight [0, :5]: {q_weight_raw[0, :5]}")
print(f"Raw Q weight [:5, 0]: {q_weight_raw[:5, 0]}")
