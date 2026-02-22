#!/usr/bin/env python3
"""Compare our Q4_K dequantization with manual calculation."""
import sys
import numpy as np

# Direct imports avoiding heavy transformers dependency
sys.path.insert(0, '/Users/redfoxhotels/zllm')
from zllm.engine.gguf_parser import GGUFParser  # Light import
from zllm.engine.quantization import dequantize_q4_k  # Our implementation

parser = GGUFParser('/Users/redfoxhotels/zllm/models/tinyllama-1.1b-q4_k_m.gguf')

# Get raw bytes for the tensor
raw = parser.load_tensor_raw('blk.0.attn_q.weight')
print(f'Raw data len: {len(raw)}')

# First block (144 bytes)
block = raw[:144]
d = np.frombuffer(block[0:2], dtype=np.float16)[0]
dmin = np.frombuffer(block[2:4], dtype=np.float16)[0]
scales = np.frombuffer(block[4:16], dtype=np.uint8)
qs = np.frombuffer(block[16:144], dtype=np.uint8)

print(f'Block 0: d={float(d):.6f}, dmin={float(dmin):.6f}')
print(f'Scales: {scales}')
print(f'qs[0:10]: {qs[:10]}')

# Our dequantization result
tensor = parser.load_tensor('blk.0.attn_q.weight')
print(f'\n--- Our dequantization ---')
print(f'First 10 values: {tensor.flatten()[:10].tolist()}')

# Manual calculation for comparison
print(f'\n--- Manual calculation ---')
# Q4_K format: 256 values per block, 4 groups of 64

# Extract scales using get_scale_min_k4 logic
sc = np.zeros(8, dtype=np.float32)
mins = np.zeros(8, dtype=np.float32)

# j < 4
sc[0:4] = scales[0:4] & 63
mins[0:4] = scales[4:8] & 63
# j >= 4  
sc[4:8] = (scales[8:12] & 0xF) | ((scales[0:4] >> 6) << 4)
mins[4:8] = (scales[8:12] >> 4) | ((scales[4:8] >> 6) << 4)

print(f'sc: {sc}')
print(f'mins: {mins}')

# Unpack 4-bit quantized values
q_lo = (qs & 0x0F).astype(np.float32)  # lower 4 bits
q_hi = ((qs >> 4) & 0x0F).astype(np.float32)  # upper 4 bits

print(f'q_lo[0:10]: {q_lo[:10]}')
print(f'q_hi[0:10]: {q_hi[:10]}')

# Manually dequantize first few values
# Group 0 (values 0-63): is1=0, is2=1
# Values 0-31: d * sc[0] * q_lo[0:32] - dmin * mins[0]
# Values 32-63: d * sc[1] * q_hi[0:32] - dmin * mins[1]

d_f = float(d)
dmin_f = float(dmin)

# Value 0: d * sc[0] * q_lo[0] - dmin * mins[0]
val0 = d_f * sc[0] * q_lo[0] - dmin_f * mins[0]
print(f'\nValue 0: d * sc[0] * q_lo[0] - dmin * mins[0]')
print(f'       = {d_f} * {sc[0]} * {q_lo[0]} - {dmin_f} * {mins[0]}')
print(f'       = {val0:.6f}')
print(f'Our result: {tensor.flatten()[0].item():.6f}')

# Value 1
val1 = d_f * sc[0] * q_lo[1] - dmin_f * mins[0]
print(f'\nValue 1: {d_f} * {sc[0]} * {q_lo[1]} - {dmin_f} * {mins[0]} = {val1:.6f}')
print(f'Our result: {tensor.flatten()[1].item():.6f}')

# Value 32 (uses q_hi with scale sc[1])
val32 = d_f * sc[1] * q_hi[0] - dmin_f * mins[1]
print(f'\nValue 32: d * sc[1] * q_hi[0] - dmin * mins[1]')
print(f'        = {d_f} * {sc[1]} * {q_hi[0]} - {dmin_f} * {mins[1]}')
print(f'        = {val32:.6f}')
print(f'Our result: {tensor.flatten()[32].item():.6f}')

# Check if our formula in quantization.py matches
# Our code: result[:, y_off:y_off+32] = d1 * q_lo[:, q_off:q_off+32] - dm1
# This is: d * sc[is1] * q_lo - dmin * mins[is1]
# BUT wait - we have:
#   d1 = (d * sc[:, is1])[:, None]
#   dm1 = (dmin * mins[:, is1])[:, None]
#   result = d1 * q_lo - dm1
# This is: (d * sc) * q_lo - (dmin * mins)
# = d * sc * q_lo - dmin * mins
# That looks right!

# But wait - in GGML's ggml-quants.c, the formula might be different
# Let me check what llama.cpp uses...
# From ggml-quants.c: *y++ = dall * (x[l] * sc - m);
# where: dall = d, m = dmin * mi, sc = scale
# So: y = d * (x * sc - m*mi/d) ... hmm
# Actually: dall = block d value
# m = dmin * mins[is]
# So: y = dall * x * sc - m = d * x * sc - dmin * mins[is]
# Our formula is: d * sc * x - dmin * mins
# Same! So formula is correct

print('\n--- Checking ggml.c formula ---')
# ggml.c Q4_K dequant:
# const float dall = GGML_FP16_TO_FP32(x[i].d);
# const float dmin = GGML_FP16_TO_FP32(x[i].dmin);
# for (int l = 0; l < 32; ++l) {
#     y[l+ 0] = dall * (q[l] & 0xF) * (sc & 0xF) - dmin * (m & 0xF);
#     y[l+32] = dall * (q[l] >>  4) * (sc >>  4) - dmin * (m >>  4);
# }
# Wait, there's additional (sc & 0xF) and (m & 0xF) masking!
# But that's for a different coding - let me look more carefully

# Actually looking at llama.cpp ggml-quants.c more carefully:
# The scales are in a different format. Let me trace through exactly.
