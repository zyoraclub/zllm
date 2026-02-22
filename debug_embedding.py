"""Compare embeddings between zllm and reference."""

import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import io, contextlib

from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf
from zllm.engine.quantization import dequantize_tensor

MODEL_PATH = 'models/tinyllama-1.1b-q4_k_m.gguf'

print("="*60)
print("EMBEDDING COMPARISON")
print("="*60)

# Load our parser
parser = GGUFParser(MODEL_PATH)
tokenizer = load_tokenizer_from_gguf(parser)

# Get embedding tensor info
emb_info = parser.tensors['token_embd.weight']
print(f"Embedding tensor: shape={emb_info.shape}, dtype={emb_info.dtype}")
print(f"GGUF shape interpretation: (hidden={emb_info.shape[0]}, vocab={emb_info.shape[1]})")

# Dequantize embedding
raw_data = parser.load_tensor_raw('token_embd.weight')
emb_weights = dequantize_tensor(raw_data, emb_info.dtype.value, emb_info.shape)
print(f"Dequantized shape: {emb_weights.shape}")

# GGUF stores (hidden, vocab), need to transpose for embedding lookup
emb_weights = emb_weights.t()  # Now (vocab, hidden)
print(f"After transpose: {emb_weights.shape}")

# Test token
test_token = 15043  # "▁Hello"
print(f"\nTest token {test_token}: '{tokenizer.vocab[test_token]}'")

# Our embedding for this token
our_emb = emb_weights[test_token].float()
print(f"\nOur embedding:")
print(f"  Shape: {our_emb.shape}")
print(f"  Mean: {our_emb.mean():.6f}")
print(f"  Std: {our_emb.std():.6f}")
print(f"  Range: [{our_emb.min():.4f}, {our_emb.max():.4f}]")
print(f"  First 10 values: {our_emb[:10].tolist()}")

# Try comparing with HuggingFace if available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n" + "="*60)
    print("HUGGINGFACE REFERENCE")
    print("="*60)
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Get HF embedding
    hf_emb = hf_model.model.embed_tokens.weight[test_token].float()
    print(f"HF embedding for token {test_token}:")
    print(f"  Mean: {hf_emb.mean():.6f}")
    print(f"  Std: {hf_emb.std():.6f}")
    print(f"  Range: [{hf_emb.min():.4f}, {hf_emb.max():.4f}]")
    print(f"  First 10 values: {hf_emb[:10].tolist()}")
    
    # Compare
    diff = (our_emb - hf_emb).abs()
    print(f"\nDifference:")
    print(f"  Max abs diff: {diff.max():.6f}")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Correlated: {torch.corrcoef(torch.stack([our_emb, hf_emb]))[0,1]:.4f}")
    
except ImportError:
    print("\nTransformers not available for HF comparison")
except Exception as e:
    print(f"\nHF comparison error: {e}")

# Test with simple forward pass through embedding
print("\n" + "="*60)
print("FORWARD TEST")
print("="*60)

with contextlib.redirect_stdout(io.StringIO()):
    engine = ZLLMInferenceEngine(MODEL_PATH)

# Get embedding from our engine
input_ids = torch.tensor([[test_token]], device=engine.device)
embed_out = engine.embed_tokens(input_ids)
print(f"Engine embedding output shape: {embed_out.shape}")
print(f"Engine embedding values:")
print(f"  Mean: {embed_out[0, 0].float().mean():.6f}")
print(f"  First 10: {embed_out[0, 0, :10].tolist()}")

# Compare with raw dequantized
raw_match = torch.allclose(
    embed_out[0, 0].float().cpu(), 
    our_emb.float(), 
    atol=1e-3
)
print(f"\nEmbedding lookup matches raw dequant: {raw_match}")
