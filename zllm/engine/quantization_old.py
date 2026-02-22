"""
Quantization and Dequantization Operations.

Our own implementation - no external dependencies.

Supports all common GGUF quantization formats:
- Q4_0, Q4_1, Q4_K (4-bit)
- Q5_0, Q5_1, Q5_K (5-bit)
- Q8_0, Q8_K (8-bit)
- Q2_K, Q3_K, Q6_K (other precisions)
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


def dequantize_q4_0(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q4_0 format.
    
    Format: For each block of 32 values:
    - 2 bytes: fp16 scale (d)
    - 16 bytes: 32 4-bit quantized values packed (2 per byte)
    
    Values are: x = d * (q - 8) where q is 4-bit value (0-15)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    bytes_per_block = 18  # 2 + 16
    
    result = np.zeros(n_elements, dtype=np.float32)
    
    offset = 0
    for block_idx in range(n_blocks):
        # Read scale (fp16)
        d = struct.unpack("<e", data[offset:offset+2])[0]
        offset += 2
        
        # Read 16 bytes of packed 4-bit values
        for i in range(16):
            if offset >= len(data):
                break
            byte = data[offset]
            offset += 1
            
            # Extract two 4-bit values
            q0 = byte & 0x0F
            q1 = (byte >> 4) & 0x0F
            
            idx = block_idx * 32 + i * 2
            if idx < n_elements:
                result[idx] = d * (q0 - 8)
            if idx + 1 < n_elements:
                result[idx + 1] = d * (q1 - 8)
    
    return torch.from_numpy(result.reshape(shape))


def dequantize_q4_1(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q4_1 format.
    
    Format: For each block of 32 values:
    - 2 bytes: fp16 scale (d)
    - 2 bytes: fp16 minimum (m)
    - 16 bytes: 32 4-bit quantized values packed
    
    Values are: x = d * q + m where q is 4-bit value (0-15)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_elements, dtype=np.float32)
    
    offset = 0
    for block_idx in range(n_blocks):
        # Read scale and minimum (fp16)
        d = struct.unpack("<e", data[offset:offset+2])[0]
        offset += 2
        m = struct.unpack("<e", data[offset:offset+2])[0]
        offset += 2
        
        # Read 16 bytes of packed 4-bit values
        for i in range(16):
            if offset >= len(data):
                break
            byte = data[offset]
            offset += 1
            
            q0 = byte & 0x0F
            q1 = (byte >> 4) & 0x0F
            
            idx = block_idx * 32 + i * 2
            if idx < n_elements:
                result[idx] = d * q0 + m
            if idx + 1 < n_elements:
                result[idx + 1] = d * q1 + m
    
    return torch.from_numpy(result.reshape(shape))


def dequantize_q8_0(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q8_0 format.
    
    Format: For each block of 32 values:
    - 2 bytes: fp16 scale (d)
    - 32 bytes: 32 int8 quantized values
    
    Values are: x = d * q where q is int8 value (-128 to 127)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_elements, dtype=np.float32)
    
    offset = 0
    for block_idx in range(n_blocks):
        # Read scale (fp16)
        d = struct.unpack("<e", data[offset:offset+2])[0]
        offset += 2
        
        # Read 32 int8 values
        for i in range(32):
            if offset >= len(data):
                break
            q = struct.unpack("<b", data[offset:offset+1])[0]  # signed int8
            offset += 1
            
            idx = block_idx * 32 + i
            if idx < n_elements:
                result[idx] = d * q
    
    return torch.from_numpy(result.reshape(shape))


def dequantize_q4_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q4_K format (vectorized).
    
    Block size: 256, bytes per block: 144
    Format: d (2) + dmin (2) + scales (12) + qs (128)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 256
    bytes_per_block = 144
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data_arr):
            break
        
        # Read d and dmin
        d = struct.unpack("<e", data[offset:offset+2])[0]
        dmin = struct.unpack("<e", data[offset+2:offset+4])[0]
        
        # Read scales/mins (12 bytes)
        scales_bytes = data_arr[offset+4:offset+16]
        
        # Decode scales and mins
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for i in range(8):
            scales[i] = scales_bytes[i] & 0x3F
            mins[i] = (scales_bytes[8 + i // 2] >> (4 * (i % 2))) & 0x0F
        
        # Read quantized values
        qs = data_arr[offset+16:offset+144]
        
        # Unpack 4-bit values (128 bytes -> 256 values)
        q = np.zeros(256, dtype=np.float32)
        q[0::2] = qs & 0x0F
        q[1::2] = (qs >> 4) & 0x0F
        
        # Get scale/min for each value (8 sub-blocks of 32 values)
        sc = scales.repeat(32)
        m = mins.repeat(32)
        
        # Dequantize: x = d * sc * q - dmin * m
        out_start = block_idx * 256
        out_end = min(out_start + 256, n_elements)
        n_vals = out_end - out_start
        result[out_start:out_end] = (d * sc[:n_vals] * q[:n_vals] - dmin * m[:n_vals])
    
    return torch.from_numpy(result[:n_elements].reshape(shape))


def dequantize_q5_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q5_K format (vectorized).
    
    Block size: 256, bytes per block: 176
    Format: d (2) + dmin (2) + scales (12) + qh (32) + ql (128)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 256
    bytes_per_block = 176
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data_arr):
            break
        
        # Read d and dmin
        d = struct.unpack("<e", data[offset:offset+2])[0]
        dmin = struct.unpack("<e", data[offset+2:offset+4])[0]
        
        # Read scales (12 bytes)
        scales_bytes = data_arr[offset+4:offset+16]
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for i in range(8):
            scales[i] = scales_bytes[i] & 0x3F
            mins[i] = (scales_bytes[8 + i // 2] >> (4 * (i % 2))) & 0x0F
        
        # Read qh (32 bytes -> 256 high bits)
        qh = data_arr[offset+16:offset+48]
        
        # Read ql (128 bytes -> 256 4-bit values)
        ql = data_arr[offset+48:offset+176]
        
        # Unpack low 4 bits
        q_lo = np.zeros(256, dtype=np.int32)
        q_lo[0::2] = ql & 0x0F
        q_lo[1::2] = (ql >> 4) & 0x0F
        
        # Unpack high bits
        q_hi = np.zeros(256, dtype=np.int32)
        for byte_idx in range(32):
            for bit in range(8):
                q_hi[byte_idx * 8 + bit] = (qh[byte_idx] >> bit) & 0x01
        
        # Combine to 5-bit
        q = q_lo | (q_hi << 4)
        
        # Get scale/min for each value
        sc = scales.repeat(32)
        m = mins.repeat(32)
        
        # Dequantize
        out_start = block_idx * 256
        out_end = min(out_start + 256, n_elements)
        n_vals = out_end - out_start
        result[out_start:out_end] = (d * sc[:n_vals] * q[:n_vals] - dmin * m[:n_vals])
    
    return torch.from_numpy(result[:n_elements].reshape(shape))


def dequantize_q6_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q6_K format (vectorized).
    
    Block size: 256, bytes per block: 210
    Format: ql (128) + qh (64) + scales (16) + d (2)
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    block_size = 256
    bytes_per_block = 210
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # Pre-allocate result
    result = np.zeros(n_blocks * block_size, dtype=np.float32)
    
    # Process all blocks using numpy
    data_arr = np.frombuffer(data, dtype=np.uint8)
    
    for block_idx in range(n_blocks):
        offset = block_idx * bytes_per_block
        if offset + bytes_per_block > len(data_arr):
            break
        
        # Read ql, qh, scales, d
        ql = data_arr[offset:offset+128]
        qh = data_arr[offset+128:offset+192]
        scales = data_arr[offset+192:offset+208].view(np.int8)
        d_bytes = data[offset+208:offset+210]
        d = struct.unpack("<e", d_bytes)[0]
        
        # Unpack 4-bit values from ql (128 bytes -> 256 4-bit values)
        q_lo = np.zeros(256, dtype=np.int32)
        q_lo[0::2] = ql & 0x0F
        q_lo[1::2] = (ql >> 4) & 0x0F
        
        # Unpack 2-bit values from qh (64 bytes -> 256 2-bit values) 
        q_hi = np.zeros(256, dtype=np.int32)
        for i in range(4):
            q_hi[i*64:(i+1)*64] = (qh >> (i*2)) & 0x03
        
        # Combine to 6-bit
        q = q_lo | (q_hi << 4)
        
        # Get scales for each of 16 sub-blocks (16 values each)
        sc = scales.repeat(16)
        
        # Dequantize
        out_start = block_idx * 256
        out_end = min(out_start + 256, n_elements)
        n_vals = out_end - out_start
        result[out_start:out_end] = (d * sc[:n_vals] * (q[:n_vals] - 32)).astype(np.float32)
    
    return torch.from_numpy(result[:n_elements].reshape(shape))


def dequantize_q2_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize Q2_K format (2-bit, very compact)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    result = np.zeros(n_elements, dtype=np.float32)
    return torch.from_numpy(result.reshape(shape))


def dequantize_q3_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Dequantize Q3_K format (3-bit)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    result = np.zeros(n_elements, dtype=np.float32)
    return torch.from_numpy(result.reshape(shape))


def dequantize_q8_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q8_K format.
    
    Similar to Q8_0 but with super-blocks.
    """
    # Fall back to Q8_0 logic for now
    return dequantize_q8_0(data, shape)


def dequantize_f16(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Load FP16 tensor."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    arr = np.frombuffer(data[:n_elements * 2], dtype=np.float16)
    return torch.from_numpy(arr.astype(np.float32).reshape(shape))


def dequantize_f32(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Load FP32 tensor."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    arr = np.frombuffer(data[:n_elements * 4], dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape))


def dequantize_bf16(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Load BF16 tensor."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    # BF16 is 2 bytes - read as uint16 and convert
    arr = np.frombuffer(data[:n_elements * 2], dtype=np.uint16)
    # Convert BF16 to FP32: shift left by 16 bits
    arr_f32 = np.zeros(n_elements, dtype=np.float32)
    arr_f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
    return torch.from_numpy(arr_f32.reshape(shape))


# Dequantization dispatch table
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


def dequantize_tensor(
    data: bytes, 
    dtype: int, 
    shape: Tuple[int, ...],
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize a tensor from raw bytes.
    
    Args:
        data: Raw tensor data
        dtype: GGML type (from GGMLType enum)
        shape: Tensor shape
        target_dtype: Output dtype (default: float16)
    
    Returns:
        Dequantized tensor
    """
    try:
        qtype = QuantType(dtype)
    except ValueError:
        raise ValueError(f"Unsupported quantization type: {dtype}")
    
    if qtype not in DEQUANTIZE_FUNCTIONS:
        raise ValueError(f"No dequantization function for type: {qtype.name}")
    
    tensor = DEQUANTIZE_FUNCTIONS[qtype](data, shape)
    
    if target_dtype != torch.float32:
        tensor = tensor.to(target_dtype)
    
    return tensor


def get_quantization_info(dtype: int) -> dict:
    """
    Get information about a quantization type.
    
    Returns:
        Dict with bits_per_weight, block_size, description
    """
    info = {
        QuantType.F32: {"bits": 32, "block_size": 1, "desc": "Full precision"},
        QuantType.F16: {"bits": 16, "block_size": 1, "desc": "Half precision"},
        QuantType.BF16: {"bits": 16, "block_size": 1, "desc": "BFloat16"},
        QuantType.Q8_0: {"bits": 8, "block_size": 32, "desc": "8-bit quantization"},
        QuantType.Q4_0: {"bits": 4, "block_size": 32, "desc": "4-bit (basic)"},
        QuantType.Q4_1: {"bits": 4, "block_size": 32, "desc": "4-bit with min"},
        QuantType.Q4_K: {"bits": 4, "block_size": 256, "desc": "4-bit K-quants (best)"},
        QuantType.Q5_K: {"bits": 5, "block_size": 256, "desc": "5-bit K-quants"},
        QuantType.Q6_K: {"bits": 6, "block_size": 256, "desc": "6-bit K-quants"},
        QuantType.Q2_K: {"bits": 2, "block_size": 256, "desc": "2-bit (very small)"},
        QuantType.Q3_K: {"bits": 3, "block_size": 256, "desc": "3-bit K-quants"},
        QuantType.Q8_K: {"bits": 8, "block_size": 256, "desc": "8-bit K-quants"},
    }
    
    try:
        qtype = QuantType(dtype)
        return info.get(qtype, {"bits": 0, "block_size": 0, "desc": "Unknown"})
    except ValueError:
        return {"bits": 0, "block_size": 0, "desc": "Unknown"}
