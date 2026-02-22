"""Compare our inference against llama-cpp-python reference."""

import sys
sys.path.insert(0, '.')
import torch
import numpy as np

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.inference import ZLLMInferenceEngine
from zllm.engine.tokenizer import load_tokenizer_from_gguf

MODEL_PATH = 'models/tinyllama-1.1b-q4_k_m.gguf'

def test_with_llama_cpp():
    """Test with llama-cpp-python as reference."""
    if not HAS_LLAMA_CPP:
        return
    
    print("="*60)
    print("REFERENCE: llama-cpp-python")
    print("="*60)
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=0,
        verbose=False
    )
    
    prompt = "The capital of France is"
    output = llm(prompt, max_tokens=20, temperature=0.1, echo=False)
    print(f"Prompt: {prompt}")
    print(f"Output: {output['choices'][0]['text']}")
    
    return output['choices'][0]['text']

def test_with_zllm():
    """Test with our engine."""
    import io, contextlib
    
    print("\n" + "="*60)
    print("ZLLM ENGINE")
    print("="*60)
    
    parser = GGUFParser(MODEL_PATH)
    tokenizer = load_tokenizer_from_gguf(parser)
    
    # Suppress verbose output
    with contextlib.redirect_stdout(io.StringIO()):
        engine = ZLLMInferenceEngine(MODEL_PATH)
    
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    print(f"Input tokens: {tokens}")
    
    output_tokens = engine.generate(tokens, max_new_tokens=20, temperature=0.1)
    output = tokenizer.decode(output_tokens)
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    
    return output

if __name__ == "__main__":
    ref_output = None
    if HAS_LLAMA_CPP:
        ref_output = test_with_llama_cpp()
    
    zllm_output = test_with_zllm()
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    if ref_output:
        print(f"Reference: {ref_output}")
    print(f"ZLLM:      {zllm_output}")
