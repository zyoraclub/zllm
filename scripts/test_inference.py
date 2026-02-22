#!/usr/bin/env python3
"""Quick inference test for debugging."""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PYTORCH_ENABLE_LAZY_TENSOR'] = '0'

import torch
from zllm.engine.inference import NativeGGUFInference, InferenceConfig

MODEL_PATH = os.path.expanduser(
    '~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/'
    'snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
)

def main():
    print("Loading model...")
    config = InferenceConfig(
        max_seq_len=256,
        device="cpu",
        dtype=torch.float32,
    )
    
    engine = NativeGGUFInference(MODEL_PATH, config)
    
    print("\nTrying inference...")
    prompt = "Hello"
    
    try:
        output = engine.generate(prompt, max_new_tokens=5)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
