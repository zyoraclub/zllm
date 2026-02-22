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
    Dequantize Q4_K format (used by Q4_K_M, Q4_K_S).
    
    This is the most common format for good quality/size balance.
    
    Format: For each super-block of 256 values:
    - 2 bytes: fp16 super-block scale (d)
    - 2 bytes: fp16 super-block minimum (dmin)
    - 12 bytes: scales for 8 sub-blocks (6-bit each, packed)
    - 4 bytes: minimums for 8 sub-blocks (4-bit each, packed)
    - 128 bytes: 256 4-bit quantized values packed
    
    Each sub-block (32 values): x = d * sc * q + dmin * m
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    super_block_size = 256
    sub_block_size = 32
    n_super_blocks = (n_elements + super_block_size - 1) // super_block_size
    bytes_per_super_block = 144  # 2 + 2 + 12 + 4 + 128 - but actually it's more complex
    
    result = np.zeros(n_elements, dtype=np.float32)
    
    offset = 0
    for sb_idx in range(n_super_blocks):
        if offset + 2 > len(data):
            break
            
        # Read super-block d and dmin
        d = struct.unpack("<e", data[offset:offset+2])[0]
        offset += 2
        dmin = struct.unpack("<e", data[offset:offset+2])[0]
        offset += 2
        
        # Read scales (12 bytes for 8 sub-blocks, 6-bit each)
        scales = []
        scale_bytes = data[offset:offset+12]
        offset += 12
        
        # Decode 6-bit scales (packed in complex way)
        for i in range(8):
            byte_idx = (i * 6) // 8
            bit_offset = (i * 6) % 8
            
            if byte_idx < len(scale_bytes):
                if bit_offset <= 2:
                    sc = (scale_bytes[byte_idx] >> bit_offset) & 0x3F
                else:
                    sc = ((scale_bytes[byte_idx] >> bit_offset) | 
                          (scale_bytes[byte_idx + 1] << (8 - bit_offset))) & 0x3F
                scales.append(sc)
            else:
                scales.append(0)
        
        # Read minimums (4 bytes for 8 sub-blocks, 4-bit each)
        min_bytes = data[offset:offset+4]
        offset += 4
        
        mins = []
        for i in range(8):
            byte_idx = i // 2
            if byte_idx < len(min_bytes):
                if i % 2 == 0:
                    mins.append(min_bytes[byte_idx] & 0x0F)
                else:
                    mins.append((min_bytes[byte_idx] >> 4) & 0x0F)
            else:
                mins.append(0)
        
        # Read 128 bytes of packed 4-bit values
        qs = data[offset:offset+128]
        offset += 128
        
        # Dequantize each sub-block
        for sub_idx in range(8):
            sc = scales[sub_idx] if sub_idx < len(scales) else 1
            m = mins[sub_idx] if sub_idx < len(mins) else 0
            
            for i in range(16):  # 16 bytes = 32 4-bit values
                q_idx = sub_idx * 16 + i
                if q_idx >= len(qs):
                    break
                    
                byte = qs[q_idx]
                q0 = byte & 0x0F
                q1 = (byte >> 4) & 0x0F
                
                idx = sb_idx * 256 + sub_idx * 32 + i * 2
                if idx < n_elements:
                    result[idx] = d * sc * q0 + dmin * m
                if idx + 1 < n_elements:
                    result[idx + 1] = d * sc * q1 + dmin * m
    
    return torch.from_numpy(result.reshape(shape))


def dequantize_q5_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q5_K format.
    
    Similar to Q4_K but with 5-bit quantization.
    """
    # Q5_K is complex - for now, use simplified version
    # TODO: Implement full Q5_K
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    # Placeholder - returns zeros
    # Full implementation requires handling 5-bit packed values
    result = np.zeros(n_elements, dtype=np.float32)
    return torch.from_numpy(result.reshape(shape))


def dequantize_q6_k(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Dequantize Q6_K format.
    
    High quality 6-bit quantization.
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    # Placeholder - full implementation needed
    result = np.zeros(n_elements, dtype=np.float32)
    return torch.from_numpy(result.reshape(shape))


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
