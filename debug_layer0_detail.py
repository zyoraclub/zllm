#!/usr/bin/env python3
"""Debug by comparing intermediate computations step by step."""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine, precompute_rope_cache, apply_rope
from zllm.engine.tokenizer import load_tokenizer_from_gguf
import io, contextlib

print("=== Loading engine ===")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)

# Test prompt
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
print(f"Tokens: {tokens}")

# Get input embeddings
input_ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
hidden = engine.embed_tokens(input_ids)
print(f"\nEmbedding output shape: {hidden.shape}")
print(f"Embedding output stats: mean={hidden.mean():.6f}, std={hidden.std():.6f}")
print(f"Embedding [:, 0, :5]: {hidden[0, 0, :5]}")

# Load layer 0
layer = engine._load_layer(0)
print(f"\n=== Layer 0 Analysis ===")

# RMSNorm
normed = layer.input_layernorm(hidden)
print(f"After input_layernorm: mean={normed.mean():.6f}, std={normed.std():.6f}")
print(f"Normed [:, 0, :5]: {normed[0, 0, :5]}")

# QKV projections
q = layer.q_proj(normed)
k = layer.k_proj(normed)
v = layer.v_proj(normed)
print(f"\nQ projection shape: {q.shape}, mean={q.mean():.6f}, std={q.std():.6f}")
print(f"K projection shape: {k.shape}, mean={k.mean():.6f}, std={k.std():.6f}")
print(f"V projection shape: {v.shape}, mean={v.mean():.6f}, std={v.std():.6f}")
print(f"Q [:, 0, :5]: {q[0, 0, :5]}")

# Reshape for attention
batch_size, seq_len = input_ids.shape
q_reshaped = q.view(batch_size, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
k_reshaped = k.view(batch_size, seq_len, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
v_reshaped = v.view(batch_size, seq_len, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
print(f"\nQ reshaped: {q_reshaped.shape}")  # [B, num_heads, T, head_dim]
print(f"K reshaped: {k_reshaped.shape}")  # [B, num_kv_heads, T, head_dim]

# RoPE
position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
cos = engine.cos_cache
sin = engine.sin_cache
print(f"\nRoPE cos cache shape: {cos.shape}")
print(f"RoPE sin cache shape: {sin.shape}")
print(f"Position IDs: {position_ids}")

q_rope, k_rope = apply_rope(q_reshaped, k_reshaped, cos, sin, position_ids)
print(f"\nAfter RoPE Q: mean={q_rope.mean():.6f}, std={q_rope.std():.6f}")
print(f"After RoPE K: mean={k_rope.mean():.6f}, std={k_rope.std():.6f}")
print(f"Q_rope [0, 0, 0, :5]: {q_rope[0, 0, 0, :5]}")

# GQA repeat
n_rep = layer.num_heads // layer.num_kv_heads
k_rep = k_rope.repeat_interleave(n_rep, dim=1)
v_rep = v_reshaped.repeat_interleave(n_rep, dim=1)
print(f"\nAfter GQA repeat K: {k_rep.shape}")

# Attention scores
import math
scale = 1.0 / math.sqrt(layer.head_dim)
attn_weights = torch.matmul(q_rope, k_rep.transpose(-2, -1)) * scale
print(f"\nAttn weights shape: {attn_weights.shape}")
print(f"Attn weights (pre-softmax): min={attn_weights.min():.4f}, max={attn_weights.max():.4f}, mean={attn_weights.mean():.6f}")

# Apply causal mask
import torch.nn.functional as F
# For prefill, we need a causal mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=engine.device), diagonal=1).bool()
attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
print(f"Attn weights (after mask): min={attn_weights[attn_weights > -1e9].min():.4f}, max={attn_weights.max():.4f}")

# Softmax
attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_rope.dtype)
print(f"Attn probs: min={attn_probs.min():.4f}, max={attn_probs.max():.4f}, sum per row={attn_probs[0,0,0,:].sum():.4f}")

# Attention output
attn_out = torch.matmul(attn_probs, v_rep)
print(f"\nAttn output shape: {attn_out.shape}")
print(f"Attn output: mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")

# Reshape and output projection
attn_out_flat = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
o_out = layer.o_proj(attn_out_flat.to(layer.o_proj.weight.dtype))
print(f"\nO projection output: mean={o_out.mean():.6f}, std={o_out.std():.6f}")

# Residual
hidden_after_attn = hidden + o_out
print(f"After residual: mean={hidden_after_attn.mean():.6f}, std={hidden_after_attn.std():.6f}")

# FFN
normed2 = layer.post_attention_layernorm(hidden_after_attn)
gate = F.silu(layer.gate_proj(normed2.to(layer.gate_proj.weight.dtype)))
up = layer.up_proj(normed2.to(layer.up_proj.weight.dtype))
ffn_out = layer.down_proj(gate * up)
print(f"\nFFN gate: mean={gate.mean():.6f}, std={gate.std():.6f}")
print(f"FFN up: mean={up.mean():.6f}, std={up.std():.6f}")
print(f"FFN down: mean={ffn_out.mean():.6f}, std={ffn_out.std():.6f}")

# Final
hidden_final = hidden_after_attn + ffn_out
print(f"\nLayer 0 output: mean={hidden_final.mean():.6f}, std={hidden_final.std():.6f}")

# Check what token 5099 (>>) has as embedding vs what model predicts
print(f"\n=== Token Analysis ===")
print(f"Token 5099 (>>) embedding: {engine.embed_tokens.weight[5099, :5]}")
print(f"Token 3681 embedding: {engine.embed_tokens.weight[3681, :5]}")

# What does the tokenizer say "Paris" is?
paris_tokens = tokenizer.encode("Paris")
print(f"'Paris' tokenizes to: {paris_tokens}")
for t in paris_tokens:
    if t != 1:  # skip BOS
        print(f"  Token {t}: '{tokenizer.decode([t])}'")
