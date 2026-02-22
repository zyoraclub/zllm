#!/usr/bin/env python3
"""Debug attention patterns - check if GQA is causing issues."""
import torch
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, '.')

from zllm.engine.inference import ZLLMInferenceEngine, apply_rope
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.tokenizer import load_tokenizer_from_gguf
import io, contextlib

print("=== Loading engine ===")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)

# Simple test
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
print(f"Tokens: {tokens}")

input_ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
hidden = engine.embed_tokens(input_ids)
batch_size, seq_len = input_ids.shape

layer = engine._load_layer(0)

# Get QKV
normed = layer.input_layernorm(hidden)
q = layer.q_proj(normed)
k = layer.k_proj(normed)
v = layer.v_proj(normed)

# Reshape
q = q.view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
k = k.view(batch_size, seq_len, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
v = v.view(batch_size, seq_len, layer.num_kv_heads, layer.head_dim).transpose(1, 2)

# RoPE
position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
q, k = apply_rope(q, k, engine.cos_cache, engine.sin_cache, position_ids)

# GQA repeat
n_rep = layer.num_heads // layer.num_kv_heads
print(f"\nGQA: {layer.num_heads} Q heads, {layer.num_kv_heads} KV heads, n_rep={n_rep}")
k_rep = k.repeat_interleave(n_rep, dim=1)
v_rep = v.repeat_interleave(n_rep, dim=1)

# Attention
scale = 1.0 / math.sqrt(layer.head_dim)
attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * scale

print(f"\n=== Pre-softmax attention weights (head 0) ===")
print(attn_weights[0, 0].detach().cpu().numpy())

# Apply causal mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=engine.device), diagonal=1).bool()
attn_weights_masked = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

# Softmax
attn_probs = F.softmax(attn_weights_masked, dim=-1, dtype=torch.float32)

print(f"\n=== Attention probs (head 0) ===")
print(attn_probs[0, 0].detach().cpu().numpy())

# Compute entropy per row
eps = 1e-9
entropy = -(attn_probs * torch.log(attn_probs + eps)).sum(-1)
print(f"\n=== Attention entropy per position (head 0) ===")
print(f"Row entropies: {entropy[0, 0].detach().cpu().numpy()}")
print(f"Mean entropy: {entropy.mean():.4f}")
print(f"Expected uniform entropy for seq_len={seq_len}: {math.log(seq_len):.4f}")

# Check if uniformity
print(f"\n=== Uniformity check ===")
for pos in range(seq_len):
    valid_positions = pos + 1  # causal: can only attend to 0..pos
    expected_uniform = 1.0 / valid_positions
    actual_max = attn_probs[0, 0, pos, :pos+1].max().item()
    actual_min = attn_probs[0, 0, pos, :pos+1].min().item()
    print(f"Pos {pos}: valid={valid_positions}, uniform={expected_uniform:.3f}, actual range=[{actual_min:.3f}, {actual_max:.3f}]")

# Compare GQA head mapping
print(f"\n=== GQA Head Mapping Check ===")
print("Q head -> KV head mapping:")
for qh in range(min(16, layer.num_heads)):
    kvh = qh // n_rep
    print(f"  Q head {qh} -> KV head {kvh}")

# Check if different Q heads attending to same KV head produce similar patterns
print(f"\n=== Cross-head similarity (Q heads sharing KV head 0) ===")
for qh in range(n_rep):  # These should all use KV head 0
    pattern = attn_probs[0, qh, -1, :].detach().cpu().numpy()  # Last position attention
    print(f"Q head {qh}: {pattern}")
