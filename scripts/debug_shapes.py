#!/usr/bin/env python3
"""Debug tensor shape flow."""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.quantization import dequantize_tensor

MODEL_PATH = os.path.expanduser(
    '~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/'
    'snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
)

def main():
    print("Loading GGUF file...")
    parser = GGUFParser(MODEL_PATH)
    weights = parser.get_weights()
    
    names = ['blk.0.attn_k.weight', 'blk.0.attn_q.weight', 
             'blk.0.attn_v.weight', 'blk.0.ffn_gate.weight']
    
    for name in names:
        if name in weights:
            info = weights[name]
            print(f'\n{name}:')
            print(f'  GGUF shape: {info["shape"]}')
            print(f'  GGUF type: {info["type"]}')
            
            # Dequantize
            tensor = dequantize_tensor(info['data'], info['type'], info['shape'])
            print(f'  Tensor shape after dequant: {tensor.shape}')
            
            # What PyTorch Linear expects
            # For k_proj with num_kv_heads=4, head_dim=64: out_features=256
            # Input is hidden_size=2048
            # Linear weight shape should be (out_features, in_features) = (256, 2048)
            
            # What GGUF likely stores: (in_features, out_features) = (2048, 256)
            # So we need to transpose!
            if list(tensor.shape) == list(info["shape"]):
                print(f'  Transposed shape: {tuple(reversed(tensor.shape))}')
    
    print("\nDone!")

if __name__ == "__main__":
    main()
