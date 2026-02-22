#!/usr/bin/env python3
"""Test different Q/K reshape patterns to find correct head layout."""
import torch
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, '.')

from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.quantization import dequantize_tensor
from zllm.engine.inference import ZLLMInferenceEngine, precompute_rope_cache, apply_rope
from zllm.engine.tokenizer import load_tokenizer_from_gguf
from gguf import GGUFReader
import io, contextlib

print("=== Setup ===")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)
reader = GGUFReader('models/tinyllama-1.1b-q4_k_m.gguf')

# Config
num_heads = 32
num_kv_heads = 4
head_dim = 64
hidden = 2048

def load_weight(name):
    for t in reader.tensors:
        if t.name == name:
            w = dequantize_tensor(bytes(t.data), t.tensor_type.value, tuple(t.shape))
            return w.to(engine.device)
    raise KeyError(name)

# Test input
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
batch_size, seq_len = input_ids.shape

# Get embeddings and normalize
hidden_states = engine.embed_tokens(input_ids)
layer = engine._load_layer(0)
normed = layer.input_layernorm(hidden_states)
normed = normed.to(torch.float16)

# Load raw weights
q_weight = load_weight('blk.0.attn_q.weight').t()  # Transpose as usual
k_weight = load_weight('blk.0.attn_k.weight').t()

# Compute Q and K
q_flat = F.linear(normed, q_weight)  # [1, 6, 2048]
k_flat = F.linear(normed, k_weight)  # [1, 6, 256]

print(f"Q flat: {q_flat.shape}, K flat: {k_flat.shape}")

def test_attention(q_reshaped, k_reshaped, name):
    """Test attention with given Q/K reshape."""
    # RoPE
    cos, sin = precompute_rope_cache(head_dim, 2048, 10000.0, engine.device)
    position_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
    q_rope, k_rope = apply_rope(q_reshaped.to(torch.float16), k_reshaped.to(torch.float16), cos, sin, position_ids)
    
    # GQA expand
    k_expanded = k_rope.repeat_interleave(num_heads // num_kv_heads, dim=1)
    
    # Attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_rope, k_expanded.transpose(-2, -1)) * scale
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=engine.device), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    probs = F.softmax(scores, dim=-1)
    
    score_range = scores[scores > -1e9]
    last_attn = probs[0, 0, -1, :].detach().cpu().numpy()
    
    print(f"\n{name}:")
    print(f"  Score range: [{score_range.min():.4f}, {score_range.max():.4f}]")
    print(f"  Last pos attn: {last_attn}")
    
    # Check if attention is peaked (not uniform)
    attn_max = probs[0, :, -1, :].max(dim=-1).values.mean().item()
    print(f"  Mean max attention: {attn_max:.4f} (uniform would be ~{1/seq_len:.4f})")

# Pattern 1: Standard sequential heads - [batch, seq, num_heads, head_dim]
print("\n=== Testing Reshape Patterns ===")

q1 = q_flat.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k1 = k_flat.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
test_attention(q1, k1, "Pattern 1: Sequential [B, T, H, D] -> [B, H, T, D]")

# Pattern 2: Interleaved heads - [batch, seq, head_dim, num_heads]
q2 = q_flat.view(batch_size, seq_len, head_dim, num_heads).permute(0, 3, 1, 2)
k2 = k_flat.view(batch_size, seq_len, head_dim, num_kv_heads).permute(0, 3, 1, 2)
test_attention(q2, k2, "Pattern 2: Interleaved [B, T, D, H] -> [B, H, T, D]")

# Pattern 3: What if heads are in reverse order?
q3 = q_flat.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).flip(dims=[1])
k3 = k_flat.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2).flip(dims=[1])
test_attention(q3, k3, "Pattern 3: Reversed heads")

# Pattern 4: Without transpose (keep GGUF layout)
q_weight_raw = load_weight('blk.0.attn_q.weight')  # No transpose
k_weight_raw = load_weight('blk.0.attn_k.weight')
q_flat_raw = torch.matmul(normed, q_weight_raw)
k_flat_raw = torch.matmul(normed, k_weight_raw)

q4 = q_flat_raw.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k4 = k_flat_raw.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
test_attention(q4, k4, "Pattern 4: No weight transpose")

# What does calling llama.cpp give?
print("\n=== llama.cpp Reference ===")
from llama_cpp import Llama
llm = Llama(model_path='models/tinyllama-1.1b-q4_k_m.gguf', n_ctx=64, verbose=False)
output = llm(prompt, max_tokens=5, echo=False, temperature=0.0)
print(f"llama.cpp output: '{output['choices'][0]['text']}'")
