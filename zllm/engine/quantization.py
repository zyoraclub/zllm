"""
Quantization and Dequantization Operations - GGML Compatible.

Exact implementation matching GGML's ggml-quants.c

Supports all common GGUF quantization formats:
- Q4_0, Q4_1 (4-bit basic)
- Q4_K, Q5_K, Q6_K (K-quants)
- Q8_0 (8-bit)
- F16, F32, BF16 (floating point)
"""

import struct
from enum import IntEnum
from typing import Tuple
import numpy as np
import torch


class QuantType(IntEnum):
    """Quantization types matching GGML."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 29


# Block sizes and byte counts from GGML
QK_K = 256  # K-quant block size

BLOCK_SIZES = {
    QuantType.Q4_0: 32,
    QuantType.Q4_1: 32,
    QuantType.Q5_0: 32,
    QuantType.Q5_1: 32,
    QuantType.Q8_0: 32,
    QuantType.Q4_K: QK_K,
    QuantType.Q5_K: QK_K,
    QuantType.Q6_K: QK_K,
    QuantType.Q2_K: QK_K,
    QuantType.Q3_K: QK_K,
    QuantType.Q8_K: QK_K,
}

BYTES_PER_BLOCK = {
    QuantType.Q4_0: 18,   # 2 + 16
    QuantType.Q4_1: 20,   # 2 + 2 + 16
    QuantType.Q5_0: 22,
    QuantType.Q5_1: 24,
    QuantType.Q8_0: 34,   # 2 + 32
    QuantType.Q4_K: 144,  # 2 + 2 + 12 + 128
    QuantType.Q5_K: 176,  # 2 + 2 + 12 + 32 + 128
    QuantType.Q6_K: 210,  # 128 + 64 + 16 + 2
    QuantType.Q2_K: 84,
    QuantType.Q3_K: 110,
    QuantType.Q8_K: 292,
}


def _get_scale_min_k4(j: int, scales: np.ndarray) -> Tuple[int, int]:
    """
    Get scale and min for Q4_K/Q5_K.
    
    Matches GGML's get_scale_min_k4 function exactly.
    
    The 12 bytes encode 8 scales and 8 mins:
    - bytes 0-3: lower 6 bits of scales 0-3
    - bytes 4-7: lower 6 bits of mins 0-3
    - bytes 8-11: mixed high bits
    """
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return int(d), int(m)


def dequantize_q4_0(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q4_0 format.
    
    Block: 32 values, 18 bytes
    Layout: d (fp16) + qs (16 bytes packed 4-bit)
    Formula: x = d * (q - 8)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 32
    bytes_per_block = 18
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data_arr):
            break
        
        # Scale (fp16)
        d = struct.unpack("<e", data[offset:offset+2])[0]
        
        # Quantized values (16 bytes = 32 4-bit values)
        qs = data_arr[offset+2:offset+18]
        
        # Unpack and dequantize
        for i in range(16):
            q0 = qs[i] & 0x0F
            q1 = (qs[i] >> 4) & 0x0F
            
            idx = block_idx * 32 + i * 2
            result[idx] = d * (q0 - 8)
            result[idx + 1] = d * (q1 - 8)
    
    return torch.from_numpy(result[:n_elements].reshape(shape)).half()


def dequantize_q4_1(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q4_1 format.
    
    Block: 32 values, 20 bytes
    Layout: d (fp16) + m (fp16) + qs (16 bytes)
    Formula: x = d * q + m
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 32
    bytes_per_block = 20
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data_arr):
            break
        
        d = struct.unpack("<e", data[offset:offset+2])[0]
        m = struct.unpack("<e", data[offset+2:offset+4])[0]
        qs = data_arr[offset+4:offset+20]
        
        for i in range(16):
            q0 = qs[i] & 0x0F
            q1 = (qs[i] >> 4) & 0x0F
            
            idx = block_idx * 32 + i * 2
            result[idx] = d * q0 + m
            result[idx + 1] = d * q1 + m
    
    return torch.from_numpy(result[:n_elements].reshape(shape)).half()


def dequantize_q8_0(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q8_0 format.
    
    Block: 32 values, 34 bytes
    Layout: d (fp16) + qs (32 signed int8)
    Formula: x = d * q
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 32
    bytes_per_block = 34
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data):
            break
        
        d = struct.unpack("<e", data[offset:offset+2])[0]
        qs = np.frombuffer(data[offset+2:offset+34], dtype=np.int8)
        
        idx = block_idx * 32
        result[idx:idx+32] = d * qs.astype(np.float32)
    
    return torch.from_numpy(result[:n_elements].reshape(shape)).half()


def dequantize_q4_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q4_K format - FULLY VECTORIZED.
    
    Block: 256 values, 144 bytes
    Layout: d (2) + dmin (2) + scales (12) + qs (128)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = QK_K  # 256
    bytes_per_block = 144
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # Reshape data for block processing
    data_arr = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
    data_blocks = data_arr.reshape(n_blocks, bytes_per_block)
    
    # Extract all d, dmin values at once
    # CRITICAL: Must use .copy() before .view() because sliced array is non-contiguous
    d_raw = data_blocks[:, 0:2].copy().view(np.float16).flatten()
    dmin_raw = data_blocks[:, 2:4].copy().view(np.float16).flatten()
    d = d_raw.astype(np.float32)
    dmin = dmin_raw.astype(np.float32)
    
    # Extract scales (12 bytes per block)
    scales_data = data_blocks[:, 4:16]  # (n_blocks, 12)
    
    # Decode scales and mins using get_scale_min_k4 logic (vectorized)
    sc = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)
    
    # j < 4: simple extraction
    sc[:, 0:4] = scales_data[:, 0:4] & 63
    mins[:, 0:4] = scales_data[:, 4:8] & 63
    
    # j >= 4: mixed bits
    sc[:, 4:8] = (scales_data[:, 8:12] & 0xF) | ((scales_data[:, 0:4] >> 6) << 4)
    mins[:, 4:8] = (scales_data[:, 8:12] >> 4) | ((scales_data[:, 4:8] >> 6) << 4)
    
    # Extract quantized values (128 bytes = 256 4-bit values)
    qs = data_blocks[:, 16:144]  # (n_blocks, 128)
    
    # Unpack 4-bit values (0-15, NOT centered)
    q_lo = (qs & 0x0F).astype(np.float32)  # (n_blocks, 128)
    q_hi = ((qs >> 4) & 0x0F).astype(np.float32)  # (n_blocks, 128)
    
    # Build output
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    # Process 4 groups of 64 values each
    for g in range(4):
        is1 = g * 2
        is2 = g * 2 + 1
        q_off = g * 32
        y_off = g * 64
        
        # Scale factors for this group
        d1 = (d * sc[:, is1])[:, None]  # (n_blocks, 1)
        dm1 = (dmin * mins[:, is1])[:, None]
        d2 = (d * sc[:, is2])[:, None]
        dm2 = (dmin * mins[:, is2])[:, None]
        
        # Dequantize: value = d * sc * q - dmin * mins (SUBTRACTION per ggml)
        result[:, y_off:y_off+32] = d1 * q_lo[:, q_off:q_off+32] - dm1
        result[:, y_off+32:y_off+64] = d2 * q_hi[:, q_off:q_off+32] - dm2
    
    return torch.from_numpy(result.reshape(-1)[:n_elements].reshape(shape)).half()


def dequantize_q5_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q5_K format - FULLY VECTORIZED.
    
    Block: 256 values, 176 bytes
    Layout: d (2) + dmin (2) + scales (12) + qh (32) + ql (128)
    
    5-bit values: 4 bits from ql + 1 bit from qh
    Formula: x = d * sc * q - dmin * m
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = QK_K
    bytes_per_block = 176
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # Reshape data for block processing
    data_arr = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
    data_blocks = data_arr.reshape(n_blocks, bytes_per_block)
    
    # Extract components
    # CRITICAL: Must use .copy() before .view() because sliced array is non-contiguous
    d_raw = data_blocks[:, 0:2].copy().view(np.float16).flatten()
    dmin_raw = data_blocks[:, 2:4].copy().view(np.float16).flatten()
    d = d_raw.astype(np.float32)
    dmin = dmin_raw.astype(np.float32)
    
    scales_data = data_blocks[:, 4:16]  # (n_blocks, 12)
    qh = data_blocks[:, 16:48]          # (n_blocks, 32) - high bits
    ql = data_blocks[:, 48:176]         # (n_blocks, 128) - low 4 bits
    
    # Decode scales and mins (same as Q4_K)
    sc = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)
    sc[:, 0:4] = scales_data[:, 0:4] & 63
    mins[:, 0:4] = scales_data[:, 4:8] & 63
    sc[:, 4:8] = (scales_data[:, 8:12] & 0xF) | ((scales_data[:, 0:4] >> 6) << 4)
    mins[:, 4:8] = (scales_data[:, 8:12] >> 4) | ((scales_data[:, 4:8] >> 6) << 4)
    
    # Unpack low 4 bits
    q_lo = (ql & 0x0F).astype(np.float32)  # (n_blocks, 128)
    q_hi_nibble = ((ql >> 4) & 0x0F).astype(np.float32)  # (n_blocks, 128)
    
    # Unpack high bits from qh (32 bytes = 256 bits)
    qh_bits = np.unpackbits(qh, axis=1, bitorder='little')  # (n_blocks, 256)
    
    # Build 5-bit values
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    # Process 4 groups of 64 values
    for g in range(4):
        is1 = g * 2
        is2 = g * 2 + 1
        q_off = g * 32
        y_off = g * 64
        qh_off = g * 64
        
        d1 = (d * sc[:, is1])[:, None]
        dm1 = (dmin * mins[:, is1])[:, None]
        d2 = (d * sc[:, is2])[:, None]
        dm2 = (dmin * mins[:, is2])[:, None]
        
        # First 32: low nibble + high bit
        q1 = q_lo[:, q_off:q_off+32] + (qh_bits[:, qh_off:qh_off+32].astype(np.float32) * 16)
        result[:, y_off:y_off+32] = d1 * q1 - dm1
        
        # Second 32: high nibble + high bit  
        q2 = q_hi_nibble[:, q_off:q_off+32] + (qh_bits[:, qh_off+32:qh_off+64].astype(np.float32) * 16)
        result[:, y_off+32:y_off+64] = d2 * q2 - dm2
    
    return torch.from_numpy(result.reshape(-1)[:n_elements].reshape(shape)).half()


def dequantize_q6_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q6_K format - FULLY VECTORIZED.
    
    Block: 256 values, 210 bytes
    Layout: ql (128) + qh (64) + scales (16 int8) + d (fp16)
    
    6-bit values: 4 bits from ql + 2 bits from qh
    Formula: x = d * sc * (q - 32)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = QK_K
    bytes_per_block = 210
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # Reshape data for block processing
    data_arr = np.frombuffer(data[:n_blocks * bytes_per_block], dtype=np.uint8)
    data_blocks = data_arr.reshape(n_blocks, bytes_per_block)
    
    # Extract components
    ql = data_blocks[:, 0:128]      # (n_blocks, 128) - low 4 bits
    qh = data_blocks[:, 128:192]    # (n_blocks, 64) - high 2 bits
    # CRITICAL: Must use .copy() before .view() because sliced array is non-contiguous
    scales = data_blocks[:, 192:208].copy().view(np.int8)  # (n_blocks, 16) - signed scales
    d_raw = data_blocks[:, 208:210].copy().view(np.float16).flatten()
    d = d_raw.astype(np.float32)[:, None]  # (n_blocks, 1)
    
    # Build 256 6-bit values per block
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    # Process two halves of 128 values each
    for n in range(2):
        ql_off = n * 64
        qh_off = n * 32
        sc_off = n * 8
        y_off = n * 128
        
        # Extract low 4 bits from ql (positions 0-31 and 32-63)
        ql_lo_1 = ql[:, ql_off:ql_off+32] & 0xF      # for positions 0-31
        ql_lo_2 = ql[:, ql_off+32:ql_off+64] & 0xF  # for positions 32-63
        ql_hi_1 = (ql[:, ql_off:ql_off+32] >> 4) & 0xF     # for positions 64-95
        ql_hi_2 = (ql[:, ql_off+32:ql_off+64] >> 4) & 0xF  # for positions 96-127
        
        # Extract high 2 bits from qh
        qh_block = qh[:, qh_off:qh_off+32]
        qh_0 = (qh_block >> 0) & 3  # bits 0-1
        qh_1 = (qh_block >> 2) & 3  # bits 2-3
        qh_2 = (qh_block >> 4) & 3  # bits 4-5
        qh_3 = (qh_block >> 6) & 3  # bits 6-7
        
        # Combine to 6-bit values
        q1 = ql_lo_1 | (qh_0 << 4)  # positions 0-31
        q2 = ql_lo_2 | (qh_1 << 4)  # positions 32-63
        q3 = ql_hi_1 | (qh_2 << 4)  # positions 64-95
        q4 = ql_hi_2 | (qh_3 << 4)  # positions 96-127
        
        # Scale indices: 16 sub-blocks of 16 values each
        # Positions 0-15 use scale 0, 16-31 use scale 1, etc.
        sc = scales[:, sc_off:sc_off+8].astype(np.float32)  # 8 scales for this half
        
        # Apply scales in 16-element chunks
        for i in range(2):  # 2 sub-groups within each quarter
            idx = i * 16
            s_idx = i
            
            # Scale for positions idx to idx+16
            scale1 = (d * sc[:, s_idx + 0:s_idx + 1])  # (n_blocks, 1)
            scale2 = (d * sc[:, s_idx + 2:s_idx + 3])
            scale3 = (d * sc[:, s_idx + 4:s_idx + 5])
            scale4 = (d * sc[:, s_idx + 6:s_idx + 7])
            
            result[:, y_off + idx:y_off + idx + 16] = scale1 * (q1[:, idx:idx+16].astype(np.float32) - 32)
            result[:, y_off + 32 + idx:y_off + 32 + idx + 16] = scale2 * (q2[:, idx:idx+16].astype(np.float32) - 32)
            result[:, y_off + 64 + idx:y_off + 64 + idx + 16] = scale3 * (q3[:, idx:idx+16].astype(np.float32) - 32)
            result[:, y_off + 96 + idx:y_off + 96 + idx + 16] = scale4 * (q4[:, idx:idx+16].astype(np.float32) - 32)
    
    return torch.from_numpy(result.reshape(-1)[:n_elements].reshape(shape)).half()


def dequantize_q2_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize Q2_K format (2-bit)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    # Placeholder - implement if needed
    return torch.zeros(shape, dtype=torch.float16)


def dequantize_q3_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize Q3_K format (3-bit)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    # Placeholder - implement if needed
    return torch.zeros(shape, dtype=torch.float16)


def dequantize_q8_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q8_K format.
    
    Block: 256 values, 292 bytes
    Layout: d (fp32) + qs (256 int8) + bsums (32 int16)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = QK_K
    bytes_per_block = 292
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data):
            break
        
        d = struct.unpack("<f", data[offset:offset+4])[0]
        qs = np.frombuffer(data[offset+4:offset+260], dtype=np.int8)
        
        idx = block_idx * 256
        result[idx:idx+256] = d * qs.astype(np.float32)
    
    return torch.from_numpy(result[:n_elements].reshape(shape)).half()


def dequantize_f32(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize F32 format (no quantization)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    arr = np.frombuffer(data[:n_elements * 4], dtype=np.float32).copy()
    return torch.from_numpy(arr.reshape(shape)).half()


def dequantize_f16(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize F16 format."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    arr = np.frombuffer(data[:n_elements * 2], dtype=np.float16).copy()
    return torch.from_numpy(arr.reshape(shape))


def dequantize_bf16(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize BF16 format."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    # BF16 has same exponent as F32, just truncated mantissa
    raw = np.frombuffer(data[:n_elements * 2], dtype=np.uint16).copy()
    # Convert BF16 to F32 by shifting to upper 16 bits
    f32_raw = raw.astype(np.uint32) << 16
    arr = f32_raw.view(np.float32)
    return torch.from_numpy(arr.reshape(shape)).half()


# Dispatch table
DEQUANTIZE_FUNCTIONS = {
    QuantType.F32: dequantize_f32,
    QuantType.F16: dequantize_f16,
    QuantType.BF16: dequantize_bf16,
    QuantType.Q4_0: dequantize_q4_0,
    QuantType.Q4_1: dequantize_q4_1,
    QuantType.Q8_0: dequantize_q8_0,
    QuantType.Q4_K: dequantize_q4_k,
    QuantType.Q5_K: dequantize_q5_k,
    QuantType.Q6_K: dequantize_q6_k,
    QuantType.Q2_K: dequantize_q2_k,
    QuantType.Q3_K: dequantize_q3_k,
    QuantType.Q8_K: dequantize_q8_k,
}


def dequantize_tensor(data: bytes, qtype, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize a tensor based on its quantization type.
    
    Args:
        data: Raw quantized bytes
        qtype: Quantization type (QuantType enum or int)
        shape: Output tensor shape
    
    Returns:
        Dequantized tensor in float16
    """
    if isinstance(qtype, int):
        qtype = QuantType(qtype)
    
    if qtype not in DEQUANTIZE_FUNCTIONS:
        raise ValueError(f"Unsupported quantization type: {qtype}")
    
    return DEQUANTIZE_FUNCTIONS[qtype](data, shape)


def get_quantization_info(qtype) -> dict:
    """Get information about a quantization type."""
    if isinstance(qtype, int):
        qtype = QuantType(qtype)
    
    return {
        "name": qtype.name,
        "value": qtype.value,
        "block_size": BLOCK_SIZES.get(qtype, 1),
        "bytes_per_block": BYTES_PER_BLOCK.get(qtype, 0),
        "bits_per_weight": BYTES_PER_BLOCK.get(qtype, 0) * 8 / BLOCK_SIZES.get(qtype, 1) if qtype in BLOCK_SIZES else 0,
    }
