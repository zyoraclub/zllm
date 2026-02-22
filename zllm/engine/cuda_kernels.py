"""
Custom CUDA Kernels for ZLLM.

High-performance CUDA implementations for:
- Dequantization (Q4_K, Q8_0, etc.)
- RMSNorm
- RoPE (Rotary Position Embeddings)
- Fused Attention

Uses Triton for portable, high-performance GPU kernels.
Falls back to PyTorch if Triton unavailable.
"""

import torch
from typing import Optional, Tuple

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


# ============================================================================
# Dequantization Kernels
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _dequant_q8_0_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel for Q8_0 dequantization.
        
        Q8_0 format: For each block of 32 values:
        - 2 bytes: fp16 scale (d)
        - 32 bytes: int8 quantized values
        
        x = d * q
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        
        # Calculate input offset (34 bytes per block of 32 values)
        input_block_offset = pid * 34
        
        # Load scale (fp16) - first 2 bytes
        scale = tl.load(input_ptr + input_block_offset).to(tl.float32)
        
        # Load and dequantize 32 int8 values
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offsets) < n_elements
        
        # Load int8 values (after 2-byte scale)
        q_vals = tl.load(
            input_ptr + input_block_offset + 2 + offsets,
            mask=mask,
            other=0
        ).to(tl.int8).to(tl.float32)
        
        # Dequantize
        result = scale * q_vals
        
        # Store
        output_offsets = block_start + offsets
        tl.store(output_ptr + output_offsets, result, mask=mask)

    @triton.jit
    def _dequant_q4_0_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel for Q4_0 dequantization.
        
        Q4_0 format: For each block of 32 values:
        - 2 bytes: fp16 scale (d)
        - 16 bytes: 32 4-bit values packed (2 per byte)
        
        x = d * (q - 8) where q is 4-bit (0-15)
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        
        # Input offset (18 bytes per block of 32 values)
        input_block_offset = pid * 18
        
        # Load scale
        scale = tl.load(input_ptr + input_block_offset).to(tl.float32)
        
        # Load packed bytes and unpack
        for i in range(16):
            byte_offset = input_block_offset + 2 + i
            packed_byte = tl.load(input_ptr + byte_offset).to(tl.uint8)
            
            # Extract low and high 4-bit values
            q0 = (packed_byte & 0x0F).to(tl.float32)
            q1 = ((packed_byte >> 4) & 0x0F).to(tl.float32)
            
            # Dequantize: x = d * (q - 8)
            x0 = scale * (q0 - 8.0)
            x1 = scale * (q1 - 8.0)
            
            # Store
            idx0 = block_start + i * 2
            idx1 = block_start + i * 2 + 1
            
            if idx0 < n_elements:
                tl.store(output_ptr + idx0, x0)
            if idx1 < n_elements:
                tl.store(output_ptr + idx1, x1)


# ============================================================================
# RMSNorm Kernel
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _rms_norm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused RMSNorm kernel.
        
        RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
        """
        row_idx = tl.program_id(0)
        row_start = row_idx * n_cols
        
        # Compute sum of squares
        sum_sq = 0.0
        for off in range(0, n_cols, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
            sum_sq += tl.sum(x * x, axis=0)
        
        # Compute RMS
        rms = tl.sqrt(sum_sq / n_cols + eps)
        
        # Normalize and scale
        for off in range(0, n_cols, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
            w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
            out = (x / rms) * w
            tl.store(output_ptr + row_start + cols, out, mask=mask)


# ============================================================================
# RoPE Kernel
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _rope_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        seq_len,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused RoPE (Rotary Position Embedding) kernel.
        
        Applies rotation to Q and K tensors using precomputed cos/sin.
        """
        pid = tl.program_id(0)
        
        # Calculate position in sequence
        seq_idx = pid // (head_dim // 2)
        dim_idx = pid % (head_dim // 2)
        
        # Load cos and sin for this position
        cos_val = tl.load(cos_ptr + seq_idx * (head_dim // 2) + dim_idx)
        sin_val = tl.load(sin_ptr + seq_idx * (head_dim // 2) + dim_idx)
        
        # Load Q values (real and imaginary parts)
        base_idx = seq_idx * head_dim
        q_real = tl.load(q_ptr + base_idx + dim_idx * 2)
        q_imag = tl.load(q_ptr + base_idx + dim_idx * 2 + 1)
        
        # Apply rotation
        q_out_real = q_real * cos_val - q_imag * sin_val
        q_out_imag = q_real * sin_val + q_imag * cos_val
        
        # Store rotated Q
        tl.store(q_out_ptr + base_idx + dim_idx * 2, q_out_real)
        tl.store(q_out_ptr + base_idx + dim_idx * 2 + 1, q_out_imag)
        
        # Same for K
        k_real = tl.load(k_ptr + base_idx + dim_idx * 2)
        k_imag = tl.load(k_ptr + base_idx + dim_idx * 2 + 1)
        
        k_out_real = k_real * cos_val - k_imag * sin_val
        k_out_imag = k_real * sin_val + k_imag * cos_val
        
        tl.store(k_out_ptr + base_idx + dim_idx * 2, k_out_real)
        tl.store(k_out_ptr + base_idx + dim_idx * 2 + 1, k_out_imag)


# ============================================================================
# Flash Attention Kernel
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Flash Attention forward kernel.
        
        Memory-efficient attention computation using tiling.
        """
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        
        # Compute batch and head indices
        off_z = off_hz // H
        off_h = off_hz % H
        
        # Initialize pointers
        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh
        
        # Block indices
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        # Load Q block
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        
        # Initialize output accumulator
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        
        # Compute softmax scaling
        qk_scale = 1.0 / tl.sqrt(BLOCK_K * 1.0)
        
        # Loop over K, V blocks
        lo = 0
        hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
        
        for start_n in range(lo, hi, BLOCK_N):
            # Load K, V blocks
            k_ptrs = K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk
            v_ptrs = V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk
            
            k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
            v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
            
            # Compute QK^T
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            
            # Apply causal mask
            if IS_CAUSAL:
                mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
                qk = tl.where(mask, qk, float("-inf"))
            
            # Online softmax
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            # Update running max and sum
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            
            # Update accumulator
            p_scale = beta / l_i_new
            acc = acc * (alpha * l_i / l_i_new)[:, None]
            acc += tl.dot(p.to(tl.float16) * p_scale[:, None], v)
            
            # Update state
            m_i = m_i_new
            l_i = l_i_new
        
        # Store output
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


# ============================================================================
# Python Wrappers
# ============================================================================

def dequant_q8_0_cuda(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    GPU-accelerated Q8_0 dequantization.
    """
    if not TRITON_AVAILABLE:
        from .quantization import dequantize_q8_0
        return dequantize_q8_0(data, shape)
    
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    # Convert bytes to tensor
    input_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    output = torch.empty(n_elements, dtype=torch.float32, device="cuda")
    
    # Launch kernel
    n_blocks = (n_elements + 31) // 32
    _dequant_q8_0_kernel[(n_blocks,)](
        input_tensor, output, n_elements,
        BLOCK_SIZE=32,
    )
    
    return output.reshape(shape).half()


def dequant_q4_0_cuda(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    GPU-accelerated Q4_0 dequantization.
    """
    if not TRITON_AVAILABLE:
        from .quantization import dequantize_q4_0
        return dequantize_q4_0(data, shape)
    
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    input_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    output = torch.empty(n_elements, dtype=torch.float32, device="cuda")
    
    n_blocks = (n_elements + 31) // 32
    _dequant_q4_0_kernel[(n_blocks,)](
        input_tensor, output, n_elements,
        BLOCK_SIZE=32,
    )
    
    return output.reshape(shape).half()


def rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GPU-accelerated RMSNorm.
    """
    if not TRITON_AVAILABLE or not x.is_cuda:
        # Fallback to PyTorch
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * norm * weight
    
    batch_size, seq_len, hidden_size = x.shape
    x_flat = x.view(-1, hidden_size)
    output = torch.empty_like(x_flat)
    
    n_rows = x_flat.shape[0]
    
    _rms_norm_kernel[(n_rows,)](
        x_flat, weight, output,
        hidden_size, eps,
        BLOCK_SIZE=min(1024, hidden_size),
    )
    
    return output.view(batch_size, seq_len, hidden_size)


def apply_rope_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated RoPE application.
    """
    if not TRITON_AVAILABLE or not q.is_cuda:
        # Fallback to PyTorch implementation
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]
        
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        
        return q_rotated, k_rotated
    
    # Triton implementation
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    grid = (seq_len * (head_dim // 2),)
    
    _rope_kernel[grid](
        q.view(-1), k.view(-1),
        cos.view(-1), sin.view(-1),
        q_out.view(-1), k_out.view(-1),
        seq_len, head_dim,
        BLOCK_SIZE=32,
    )
    
    return q_out, k_out


def flash_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    GPU-accelerated Flash Attention.
    
    Memory-efficient attention using tiling.
    """
    if not TRITON_AVAILABLE or not q.is_cuda:
        # Fallback to PyTorch
        scale = 1.0 / (q.shape[-1] ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if causal:
            seq_len = q.shape[2]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            attn = attn.masked_fill(mask.bool(), float("-inf"))
        
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
    
    # Triton Flash Attention
    batch, n_heads, seq_len, head_dim = q.shape
    
    output = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, n_heads, seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=head_dim,
        IS_CAUSAL=causal,
    )
    
    return output


# ============================================================================
# Utility Functions
# ============================================================================

def get_cuda_capability() -> Tuple[int, int]:
    """Get CUDA compute capability."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return torch.cuda.get_device_capability(device)
    return (0, 0)


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return TRITON_AVAILABLE


def get_backend_info() -> dict:
    """Get information about available backends."""
    info = {
        "triton_available": TRITON_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_capability"] = get_cuda_capability()
        info["device_name"] = torch.cuda.get_device_name(0)
    
    return info


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_dequant(n_elements: int = 1024 * 1024, warmup: int = 10, iters: int = 100):
    """Benchmark dequantization kernels."""
    import time
    
    # Create test data
    bytes_per_block = 34  # Q8_0
    n_blocks = (n_elements + 31) // 32
    data = bytes(n_blocks * bytes_per_block)
    shape = (n_elements,)
    
    # Warmup
    for _ in range(warmup):
        _ = dequant_q8_0_cuda(data, shape)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = dequant_q8_0_cuda(data, shape)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    throughput = (n_elements * iters) / elapsed / 1e9
    print(f"Q8_0 Dequant: {throughput:.2f} Gelements/s")
    
    return throughput
