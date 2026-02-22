#!/usr/bin/env python3
"""Compare embeddings between our engine and raw GGUF."""
import numpy as np
from gguf import GGUFReader
import sys
sys.path.insert(0, '.')
from zllm.engine.quantization import dequantize_tensor
from zllm.engine.inference import ZLLMInferenceEngine
import io, contextlib

# Load using our engine
print("Loading our engine...")
with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine('models/tinyllama-1.1b-q4_k_m.gguf')

# Get embedding for token 450 ('The')
token_id = 450
our_emb = engine.embed_tokens.weight[token_id].detach().cpu().numpy()
print(f'Token 450 embedding from our engine:')
print(f'Shape: {our_emb.shape}')
print(f'Stats: mean={our_emb.mean():.6f}, std={our_emb.std():.6f}')
print(f'First 10: {our_emb[:10]}')

# Now load raw from GGUF and dequantize
print("\nLoading raw GGUF...")
reader = GGUFReader('models/tinyllama-1.1b-q4_k_m.gguf')
for t in reader.tensors:
    if t.name == 'token_embd.weight':
        raw = dequantize_tensor(bytes(t.data), t.tensor_type.value, tuple(t.shape))
        print(f'Raw GGUF embedding shape: {raw.shape}')
        
        # GGUF stores as (hidden, vocab) = (2048, 32000)
        # So token embedding is raw[:, token_id]
        if raw.shape[0] == 2048:
            ref_emb = raw[:, token_id].numpy()  # (hidden, vocab) -> get column
            print("GGUF format: (hidden, vocab) - using column indexing")
        else:
            ref_emb = raw[token_id].numpy()  # (vocab, hidden) -> get row
            print("GGUF format: (vocab, hidden) - using row indexing")
            
        print(f'Reference token 450 embedding:')
        print(f'Shape: {ref_emb.shape}')
        print(f'First 10: {ref_emb[:10]}')
        
        # Check if we match
        diff = np.abs(our_emb - ref_emb)
        print(f'\nMax diff: {diff.max():.6f}')
        print(f'Mean diff: {diff.mean():.6f}')
        print(f'Are they close? {np.allclose(our_emb, ref_emb)}')
        
        # Also check what our engine's _load_embedding does
        print(f'\n=== Engine embed_tokens shape: {engine.embed_tokens.weight.shape}')
        break
