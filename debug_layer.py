#!/usr/bin/env python3
"""Debug layer explosion by isolating attention vs FFN."""

import sys
sys.path.insert(0, '/Users/redfoxhotels/zllm')

import torch
import torch.nn.functional as F
import math
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf

print("=== Loading Model ===")
parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)
engine = ZLLMInferenceEngine('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Check RMSNorm weights
print("\n=== RMSNorm Weight Stats ===")
layer0 = engine._load_layer(0)  # Force load layer 0
print(f"input_layernorm weight: mean={layer0.input_layernorm.weight.mean():.4f}, std={layer0.input_layernorm.weight.std():.4f}, min={layer0.input_layernorm.weight.min():.4f}, max={layer0.input_layernorm.weight.max():.4f}")
print(f"post_attn_layernorm weight: mean={layer0.post_attention_layernorm.weight.mean():.4f}, std={layer0.post_attention_layernorm.weight.std():.4f}")

# Check projection weight stats
print("\n=== Layer 0 Projection Weight Stats ===")
print(f"q_proj: mean={layer0.q_proj.weight.mean():.6f}, std={layer0.q_proj.weight.std():.6f}")
print(f"k_proj: mean={layer0.k_proj.weight.mean():.6f}, std={layer0.k_proj.weight.std():.6f}")
print(f"v_proj: mean={layer0.v_proj.weight.mean():.6f}, std={layer0.v_proj.weight.std():.6f}")
print(f"o_proj: mean={layer0.o_proj.weight.mean():.6f}, std={layer0.o_proj.weight.std():.6f}")
print(f"gate_proj: mean={layer0.gate_proj.weight.mean():.6f}, std={layer0.gate_proj.weight.std():.6f}")

# Manual forward pass with detailed debugging
print("\n=== Manual Forward Pass Debug ===")
prompt = 'The capital of France is'
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], device=engine.device)
seq_len = input_ids.shape[1]
position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)

# Get embeddings
hidden_states = engine.embed_tokens(input_ids)
print(f"Embedding: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

# Manual layer 0 forward with detailed prints
residual = hidden_states
dtype = hidden_states.dtype

# RMSNorm
norm_hidden = layer0.input_layernorm(hidden_states)
print(f"After input_layernorm: mean={norm_hidden.mean():.6f}, std={norm_hidden.std():.6f}")

# QKV projections
q = layer0.q_proj(norm_hidden)
k = layer0.k_proj(norm_hidden)
v = layer0.v_proj(norm_hidden)
print(f"Q proj: mean={q.mean():.6f}, std={q.std():.6f}")
print(f"K proj: mean={k.mean():.6f}, std={k.std():.6f}")
print(f"V proj: mean={v.mean():.6f}, std={v.std():.6f}")

# Reshape
batch_size = 1
q = q.view(batch_size, seq_len, layer0.num_heads, layer0.head_dim).transpose(1, 2)
k = k.view(batch_size, seq_len, layer0.num_kv_heads, layer0.head_dim).transpose(1, 2)
v = v.view(batch_size, seq_len, layer0.num_kv_heads, layer0.head_dim).transpose(1, 2)

# Apply RoPE
from zllm.engine.inference import apply_rope
q, k = apply_rope(q, k, engine.cos_cache, engine.sin_cache, position_ids)
print(f"After RoPE Q: mean={q.mean():.6f}, std={q.std():.6f}")
print(f"After RoPE K: mean={k.mean():.6f}, std={k.std():.6f}")

# GQA repeat
if layer0.num_kv_heads < layer0.num_heads:
    n_rep = layer0.num_heads // layer0.num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    print(f"After GQA repeat - K shape: {k.shape}, V shape: {v.shape}")

# Attention scores
scale = 1.0 / math.sqrt(layer0.head_dim)
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
print(f"Attn weights (pre-softmax): mean={attn_weights.mean():.6f}, std={attn_weights.std():.6f}, min={attn_weights.min():.6f}, max={attn_weights.max():.6f}")

# Causal mask
mask = torch.full((seq_len, seq_len), float("-inf"), device=engine.device)
mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
attn_weights = attn_weights + mask

# Softmax
attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
print(f"Attn weights (post-softmax): mean={attn_weights_softmax.mean():.6f}, std={attn_weights_softmax.std():.6f}")

# Attention output
attn_out = torch.matmul(attn_weights_softmax, v)
print(f"Attn output (before o_proj): mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")

# Reshape and project
attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
attn_out = layer0.o_proj(attn_out)
print(f"Attn output (after o_proj): mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")

# Residual
hidden_states = residual + attn_out
print(f"After attention residual: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

# FFN
residual = hidden_states
hidden_states = layer0.post_attention_layernorm(hidden_states)
print(f"After post_attn_layernorm: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

gate = F.silu(layer0.gate_proj(hidden_states))
up = layer0.up_proj(hidden_states)
print(f"Gate: mean={gate.mean():.6f}, std={gate.std():.6f}")
print(f"Up: mean={up.mean():.6f}, std={up.std():.6f}")

ffn_out = layer0.down_proj(gate * up)
print(f"FFN output: mean={ffn_out.mean():.6f}, std={ffn_out.std():.6f}")

hidden_states = residual + ffn_out
print(f"After FFN residual (Layer 0 output): mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")
