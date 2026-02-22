#!/usr/bin/env python3
"""Debug lm_head and embedding orientation."""

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
llm = Llama(model_path='/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf', n_ctx=64, verbose=False, logits_all=True)

print("\n=== Weight Shapes ===")
print(f"embed_tokens.weight: {engine.embed_tokens.weight.shape}")
print(f"lm_head.weight: {engine.lm_head.weight.shape}")
print(f"final norm weight: {engine.norm.weight.shape}")

# Check GGUF shapes
print("\n=== GGUF Tensor Shapes ===")
for name in ['token_embd.weight', 'output.weight']:
    if name in parser.tensors:
        info = parser.tensors[name]
        print(f"{name}: {info.shape}")

# Get Paris token ID
paris_tokens = tokenizer.encode("Paris")
print(f"\n'Paris' encodes to: {paris_tokens}")
paris_id = paris_tokens[-1] if len(paris_tokens) > 1 else paris_tokens[0]
print(f"Paris token ID: {paris_id}")

# Run our forward
prompt = 'The capital of France is'
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens], device=engine.device)

engine.clear_kv_cache()
with torch.inference_mode():
    our_logits = engine.forward(input_ids, use_cache=False)

print(f"\n=== Our Logits ===")
our_last_logits = our_logits[0, -1, :].float().cpu().numpy()
print(f"Logits shape: {our_logits.shape}")
print(f"Logit for Paris ({paris_id}): {our_last_logits[paris_id]:.4f}")

# Get top-5 predictions
top5_ours = np.argsort(our_last_logits)[-5:][::-1]
print(f"Our top-5 token IDs: {top5_ours}")
print(f"Our top-5 tokens: {[tokenizer.decode([t]) for t in top5_ours]}")
print(f"Our top-5 logits: {[our_last_logits[t] for t in top5_ours]}")

# Run llama.cpp
print(f"\n=== llama.cpp Logits ===")
llm.reset()
llm.eval(tokens)
ref_logits = np.array(llm.scores[len(tokens)-1])  # Last position
print(f"Logit for Paris ({paris_id}): {ref_logits[paris_id]:.4f}")

top5_ref = np.argsort(ref_logits)[-5:][::-1]
print(f"Ref top-5 token IDs: {top5_ref}")
print(f"Ref top-5 tokens: {[tokenizer.decode([t]) for t in top5_ref]}")
print(f"Ref top-5 logits: {[ref_logits[t] for t in top5_ref]}")

# Compare logit distributions
print(f"\n=== Distribution Comparison ===")
print(f"Our mean: {our_last_logits.mean():.4f}, std: {our_last_logits.std():.4f}")
print(f"Ref mean: {ref_logits.mean():.4f}, std: {ref_logits.std():.4f}")

# Correlation check
corr = np.corrcoef(our_last_logits, ref_logits)[0, 1]
print(f"Correlation: {corr:.6f}")

# Check if logits are permuted (check if same tokens have high/low values)
print(f"\n=== Permutation Check ===")
# Sort both by value and see if order matches
our_order = np.argsort(our_last_logits)
ref_order = np.argsort(ref_logits)
# Check overlap in top-100
our_top100 = set(our_order[-100:])
ref_top100 = set(ref_order[-100:])
overlap = len(our_top100 & ref_top100)
print(f"Top-100 overlap: {overlap}/100")
