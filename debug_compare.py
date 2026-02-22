#!/usr/bin/env python3
"""Compare our logits with llama.cpp reference."""
import sys, io, contextlib, torch, math
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '.')
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine, apply_rope
from zllm.engine.tokenizer import load_tokenizer_from_gguf
from llama_cpp import Llama

parser = GGUFParser('models/tinyllama-1.1b-q4_k_m.gguf')
tokenizer = load_tokenizer_from_gguf(parser)

with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

prompt = 'The capital of France is'
tokens = tokenizer.encode(prompt)
print(f'Tokens: {tokens}')

# What is >> token?
gg_tok = tokenizer.encode('>>')
print(f'>> encodes to: {gg_tok}')

# Paris token
paris_tok = tokenizer.encode('Paris')
print(f'Paris encodes to: {paris_tok}')

# Run OUR forward
input_ids = torch.tensor([tokens], device=engine.device)
engine.clear_kv_cache()
with torch.inference_mode():
    our_logits = engine.forward(input_ids, use_cache=False)

last_logits = our_logits[0, -1].float().cpu()
print(f'\nOur logit for >> (id {gg_tok[-1]}): {last_logits[gg_tok[-1]]:.4f}')
print(f'Our logit for Paris (id {paris_tok[-1]}): {last_logits[paris_tok[-1]]:.4f}')

# Top 10
top_vals, top_ids = torch.topk(last_logits, 10)
print('\nOur top-10:')
for v, i in zip(top_vals.tolist(), top_ids.tolist()):
    tok = tokenizer.decode([i])
    print(f'  {i}: {repr(tok)} = {v:.4f}')

# Compare with llama.cpp
print('\n=== llama.cpp reference ===')
llm = Llama(model_path='models/tinyllama-1.1b-q4_k_m.gguf', n_ctx=64, verbose=False, logits_all=True)
llm.reset()
llm.eval(tokens)
ref_logits = np.array(llm.scores[len(tokens)-1])
print(f'Ref logit for >> (id {gg_tok[-1]}): {ref_logits[gg_tok[-1]]:.4f}')
print(f'Ref logit for Paris (id {paris_tok[-1]}): {ref_logits[paris_tok[-1]]:.4f}')

top_ref_ids = np.argsort(ref_logits)[-10:][::-1]
print('\nRef top-10:')
for i in top_ref_ids:
    print(f'  {i}: {repr(tokenizer.decode([i]))} = {ref_logits[i]:.4f}')

# Correlation
corr = np.corrcoef(last_logits.numpy(), ref_logits)[0,1]
print(f'\nLogits correlation: {corr:.4f}')
