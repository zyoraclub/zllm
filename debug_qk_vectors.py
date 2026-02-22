#!/usr/bin/env python3
"""Check if Q/K vectors vary meaningfully across positions and heads."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from zllm.engine.inference import ZLLMInferenceEngine, precompute_rope_cache, apply_rope
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.tokenizer import load_tokenizer_from_gguf
import io, contextlib

print("=== Setup ===")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)

prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
print(f"Tokens: {tokens}")
print(f"Decoded: {[tokenizer.decode([t]) for t in tokens]}")

input_ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
batch_size, seq_len = input_ids.shape

hidden = engine.embed_tokens(input_ids)
layer = engine._load_layer(0)
normed = layer.input_layernorm(hidden)

# Get Q/K
q = layer.q_proj(normed)  # [1, 6, 2048]
k = layer.k_proj(normed)  # [1, 6, 256]

print("\n=== Q/K Per-Position Stats ===")
for pos in range(seq_len):
    q_pos = q[0, pos]  # [2048]
    k_pos = k[0, pos]  # [256]
    print(f"Pos {pos}: Q mean={q_pos.mean():.4f}, std={q_pos.std():.4f} | K mean={k_pos.mean():.4f}, std={k_pos.std():.4f}")

print("\n=== Q/K Inter-Position Correlation ===")
# Check if Q vectors are similar across positions
q_flat = q[0]  # [6, 2048]
for i in range(seq_len):
    for j in range(i+1, seq_len):
        cosine_sim = F.cosine_similarity(q_flat[i:i+1], q_flat[j:j+1]).item()
        print(f"Q cos_sim(pos {i}, pos {j}): {cosine_sim:.4f}")

print("\n=== Q/K Dot Products (what attention sees) ===")
# Reshape for attention
num_heads = 32
num_kv_heads = 4
head_dim = 64

q_heads = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [1, 32, 6, 64]
k_heads = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [1, 4, 6, 64]

# For head 0, check raw dot products (before RoPE)
print("Head 0 Q@K (no RoPE, no GQA repeat):")
# Q head 0 uses K head 0 (GQA: heads 0-7 share KV head 0)
q_h0 = q_heads[0, 0]  # [6, 64]
k_h0 = k_heads[0, 0]  # [6, 64]
dots_no_rope = torch.matmul(q_h0, k_h0.T)  # [6, 6]
print(dots_no_rope.detach().cpu().numpy())

# With RoPE
cos, sin = precompute_rope_cache(head_dim, 2048, 10000.0, engine.device)
position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
q_rope, k_rope = apply_rope(q_heads, k_heads, cos, sin, position_ids)

print("\nHead 0 Q@K (with RoPE):")
q_h0_rope = q_rope[0, 0]
k_h0_rope = k_rope[0, 0]
dots_rope = torch.matmul(q_h0_rope, k_h0_rope.T)
print(dots_rope.detach().cpu().numpy())

# What are the actual vector magnitudes?
print("\n=== Vector Magnitudes ===")
print(f"Q head 0 norms per position: {q_h0.norm(dim=-1).detach().cpu().numpy()}")
print(f"K head 0 norms per position: {k_h0.norm(dim=-1).detach().cpu().numpy()}")

# Expected dot product if vectors were random with same norms
# dot ≈ norm(q) * norm(k) * cos(angle)
# For random vectors in 64D, expected cos ≈ 0
print(f"Expected max dot if aligned: {q_h0.norm(dim=-1).mean() * k_h0.norm(dim=-1).mean():.4f}")

# Let's also check what the ACTUAL Q and K values look like
print("\n=== Sample Values ===")
print(f"Q[0, head0, pos0, :8]: {q_heads[0, 0, 0, :8].detach().cpu().numpy()}")
print(f"K[0, head0, pos0, :8]: {k_heads[0, 0, 0, :8].detach().cpu().numpy()}")

# Check if dequantized weights look reasonable
print("\n=== Weight Stats ===")
print(f"Q weight: shape={layer.q_proj.weight.shape}, mean={layer.q_proj.weight.mean():.6f}, std={layer.q_proj.weight.std():.6f}")
print(f"K weight: shape={layer.k_proj.weight.shape}, mean={layer.k_proj.weight.mean():.6f}, std={layer.k_proj.weight.std():.6f}")

# Check weight distribution
q_w = layer.q_proj.weight.float()
print(f"Q weight percentiles: 1%={q_w.quantile(0.01):.4f}, 50%={q_w.quantile(0.5):.4f}, 99%={q_w.quantile(0.99):.4f}")
