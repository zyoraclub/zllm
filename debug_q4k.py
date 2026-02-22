#!/usr/bin/env python3
"""Debug Q4_K dequantization by comparing with reference."""

import sys
sys.path.insert(0, '/Users/redfoxhotels/zllm')

import numpy as np
from zllm.engine.gguf_parser import GGUFParser
from zllm.engine.quantization import dequantize_tensor

# Load our parser
parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Get a Q4_K tensor
tensor_name = 'blk.0.attn_q.weight'
info = parser.tensors[tensor_name]
raw_data = parser.load_tensor_raw(tensor_name)

print(f"Tensor: {tensor_name}")
print(f"Shape: {info.shape}, Type: {info.dtype}")
print(f"Raw data length: {len(raw_data)}")

# Extract first block (144 bytes)
block = raw_data[:144]
print(f"\n=== First Block (144 bytes) ===")
print(f"Hex dump first 32 bytes: {block[:32].hex()}")

# Parse the block manually to check our understanding
d_raw = np.frombuffer(block[0:2], dtype=np.float16)[0]
dmin_raw = np.frombuffer(block[2:4], dtype=np.float16)[0]
scales_data = np.frombuffer(block[4:16], dtype=np.uint8)
qs = np.frombuffer(block[16:144], dtype=np.uint8)

print(f"\nd (fp16): {d_raw}")
print(f"dmin (fp16): {dmin_raw}")
print(f"scales (12 bytes): {scales_data}")

# Decode sc and mins using our logic
sc = np.zeros(8, dtype=np.float32)
mins = np.zeros(8, dtype=np.float32)

# j < 4: simple extraction
sc[0:4] = scales_data[0:4] & 63
mins[0:4] = scales_data[4:8] & 63

# j >= 4: mixed bits
sc[4:8] = (scales_data[8:12] & 0xF) | ((scales_data[0:4] >> 6) << 4)
mins[4:8] = (scales_data[8:12] >> 4) | ((scales_data[4:8] >> 6) << 4)

print(f"\nDecoded sc: {sc} (range: {sc.min()}-{sc.max()})")
print(f"Decoded mins: {mins} (range: {mins.min()}-{mins.max()})")

# Unpack qs
q_lo = (qs & 0x0F).astype(np.float32)
q_hi = ((qs >> 4) & 0x0F).astype(np.float32)
print(f"\nq_lo range: {q_lo.min()}-{q_lo.max()}")
print(f"q_hi range: {q_hi.min()}-{q_hi.max()}")

# Dequant the block
d = float(d_raw)
dmin = float(dmin_raw)
result = np.zeros(256, dtype=np.float32)

for g in range(4):
    is1 = g * 2
    is2 = g * 2 + 1
    q_off = g * 32
    y_off = g * 64
    
    d1 = d * sc[is1]
    dm1 = dmin * mins[is1]
    d2 = d * sc[is2]
    dm2 = dmin * mins[is2]
    
    result[y_off:y_off+32] = d1 * q_lo[q_off:q_off+32] - dm1
    result[y_off+32:y_off+64] = d2 * q_hi[q_off:q_off+32] - dm2

print(f"\n=== Our Dequant Result (first block) ===")
print(f"Mean: {result.mean():.6f}, Std: {result.std():.6f}")
print(f"Min: {result.min():.6f}, Max: {result.max():.6f}")
print(f"First 16 values: {result[:16]}")

# Now use gguf library to get reference
try:
    from gguf import GGUFReader
    import ctypes
    
    reader = GGUFReader('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')
    
    for t in reader.tensors:
        if t.name == tensor_name:
            ref_data = bytes(t.data)
            print(f"\n=== Reference (gguf library) ===")
            print(f"Data matches: {ref_data[:144] == block}")
            break
except Exception as e:
    print(f"\nCould not load gguf reference: {e}")

# Compare with full tensor dequant
our_tensor = dequantize_tensor(raw_data, info.dtype.value, info.shape)
print(f"\n=== Full Tensor Dequant ===")
print(f"Mean: {our_tensor.float().mean():.6f}, Std: {our_tensor.float().std():.6f}")
print(f"First 16: {our_tensor.flatten()[:16].tolist()}")

# Check if block result matches full tensor
print(f"\n=== Block vs Full Tensor Match ===")
full_first_256 = our_tensor.flatten()[:256].numpy()
diff = np.abs(result - full_first_256)
print(f"Max diff: {diff.max():.6f}")
