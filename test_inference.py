#!/Users/redfoxhotels/.pyenv/versions/3.11.9/bin/python
"""Test inference after Q4_K dequantization fix."""

import sys
sys.path.insert(0, '/Users/redfoxhotels/zllm')

import torch
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf
from zllm.engine.quantization import dequantize_tensor

print("=== Weight Stats Check ===")
parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Check Q4_K weight stats
tensor_name = 'blk.0.attn_q.weight'
info = parser.tensors[tensor_name]
raw = parser.load_tensor_raw(tensor_name)
weight = dequantize_tensor(raw, info.dtype.value, info.shape)
print(f"Q weight: mean={weight.float().mean():.6f}, std={weight.float().std():.6f}")

print("\n=== Loading Model ===")
tokenizer = load_tokenizer_from_gguf(parser)
engine = ZLLMInferenceEngine('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

prompt = 'The capital of France is'
print(f"\nPrompt: {prompt}")

tokens = tokenizer.encode(prompt)
output_tokens = engine.generate(tokens, max_new_tokens=15, temperature=0.1)
output = tokenizer.decode(output_tokens)

print(f"\nOutput: {output}")
print("\nExpected output should contain 'Paris'")
