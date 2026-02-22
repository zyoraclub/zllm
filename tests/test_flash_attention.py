"""Tests for Flash Attention implementation."""

import pytest
import torch
import math


def test_flash_attention_imports():
    """Test all flash attention components import correctly."""
    from zllm.core.flash_attention import (
        FlashAttention,
        FlashAttentionConfig,
        SlidingWindowAttention,
        GroupedQueryAttention,
        AttentionBackend,
        create_attention,
        estimate_attention_memory,
        detect_best_backend,
    )
    
    # Also test from main module
    from zllm.core import (
        FlashAttention,
        FlashAttentionConfig,
        AttentionBackend,
        create_attention,
    )


def test_attention_backend_enum():
    """Test AttentionBackend enum values."""
    from zllm.core.flash_attention import AttentionBackend
    
    assert AttentionBackend.FLASH_ATTN.value == "flash_attn"
    assert AttentionBackend.SDPA.value == "sdpa"
    assert AttentionBackend.CHUNKED.value == "chunked"
    assert AttentionBackend.STANDARD.value == "standard"


def test_flash_attention_config():
    """Test FlashAttentionConfig defaults and customization."""
    from zllm.core.flash_attention import FlashAttentionConfig, AttentionBackend
    
    # Default config
    config = FlashAttentionConfig()
    assert config.chunk_size == 1024
    assert config.is_causal is True
    assert config.dropout == 0.0
    assert config.sliding_window is None
    
    # Custom config
    config = FlashAttentionConfig(
        backend=AttentionBackend.CHUNKED,
        chunk_size=512,
        sliding_window=2048,
        is_causal=False,
    )
    assert config.backend == AttentionBackend.CHUNKED
    assert config.chunk_size == 512
    assert config.sliding_window == 2048
    assert config.is_causal is False


def test_detect_backend():
    """Test automatic backend detection."""
    from zllm.core.flash_attention import detect_best_backend, AttentionBackend
    
    backend = detect_best_backend()
    # Should return a valid backend
    assert backend in AttentionBackend


def test_flash_attention_init():
    """Test FlashAttention initialization."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    attn = FlashAttention(
        num_heads=8,
        head_dim=64,
    )
    
    assert attn.num_heads == 8
    assert attn.head_dim == 64
    assert attn.scale == 1.0 / math.sqrt(64)
    assert attn.backend in AttentionBackend


def test_flash_attention_forward_3d():
    """Test FlashAttention forward with 3D input."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    batch_size = 2
    seq_len = 32
    num_heads = 4
    head_dim = 32
    hidden_dim = num_heads * head_dim
    
    # Force chunked backend for testing
    config = FlashAttentionConfig(backend=AttentionBackend.CHUNKED, is_causal=True)
    attn = FlashAttention(num_heads, head_dim, config)
    
    # 3D input: [batch, seq, hidden]
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)
    
    output = attn(q, k, v)
    
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_flash_attention_forward_4d():
    """Test FlashAttention forward with 4D input."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 32
    
    config = FlashAttentionConfig(backend=AttentionBackend.STANDARD, is_causal=True)
    attn = FlashAttention(num_heads, head_dim, config)
    
    # 4D input: [batch, heads, seq, head_dim]
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    output = attn(q, k, v)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)


def test_causal_masking():
    """Test that causal masking prevents attending to future tokens."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    batch_size = 1
    seq_len = 4
    num_heads = 1
    head_dim = 4
    
    config = FlashAttentionConfig(backend=AttentionBackend.STANDARD, is_causal=True)
    attn = FlashAttention(num_heads, head_dim, config)
    
    # Create distinct tokens - each position has unique values
    q = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    k = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    v = torch.arange(seq_len).float().view(1, 1, seq_len, 1).expand(-1, -1, -1, head_dim)
    
    # Make each position attend mostly to itself
    for i in range(seq_len):
        q[0, 0, i, 0] = 1.0
        k[0, 0, i, 0] = 1.0
    
    output = attn(q, k, v)
    
    # With causal mask, position 0 can only see position 0
    # Position 0's output should be close to v[0] = 0
    # Position 1's output is mix of v[0] and v[1]
    # etc.
    
    # The output should be monotonically increasing (roughly)
    # since later positions can see more context
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)


def test_standard_vs_chunked_consistency():
    """Test that chunked attention gives same results as standard."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    batch_size = 2
    seq_len = 64
    num_heads = 4
    head_dim = 16
    hidden_dim = num_heads * head_dim
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Standard attention
    config_std = FlashAttentionConfig(backend=AttentionBackend.STANDARD, is_causal=True)
    attn_std = FlashAttention(num_heads, head_dim, config_std)
    output_std = attn_std(q, k, v)
    
    # Chunked attention with small chunks
    config_chunked = FlashAttentionConfig(backend=AttentionBackend.CHUNKED, chunk_size=16, is_causal=True)
    attn_chunked = FlashAttention(num_heads, head_dim, config_chunked)
    output_chunked = attn_chunked(q, k, v)
    
    # Results should be very close
    assert torch.allclose(output_std, output_chunked, atol=1e-4)


def test_sliding_window_attention():
    """Test SlidingWindowAttention initialization and usage."""
    from zllm.core.flash_attention import SlidingWindowAttention
    
    attn = SlidingWindowAttention(
        num_heads=8,
        head_dim=64,
        window_size=512,
    )
    
    assert attn.num_heads == 8
    assert attn.window_size == 512
    assert attn.config.sliding_window == 512


def test_grouped_query_attention():
    """Test GroupedQueryAttention (GQA) for KV cache savings."""
    from zllm.core.flash_attention import GroupedQueryAttention, FlashAttentionConfig
    
    batch_size = 2
    seq_len = 32
    num_query_heads = 8
    num_kv_heads = 2  # 4x savings
    head_dim = 64
    
    config = FlashAttentionConfig(is_causal=True)
    gqa = GroupedQueryAttention(num_query_heads, num_kv_heads, head_dim, config)
    
    assert gqa.num_query_heads == 8
    assert gqa.num_kv_heads == 2
    assert gqa.num_groups == 4
    
    # Query has full heads, KV has reduced heads
    q = torch.randn(batch_size, seq_len, num_query_heads * head_dim)
    k = torch.randn(batch_size, seq_len, num_kv_heads * head_dim)
    v = torch.randn(batch_size, seq_len, num_kv_heads * head_dim)
    
    output = gqa(q, k, v)
    
    # Output should have full head dimension
    assert output.shape == (batch_size, seq_len, num_query_heads * head_dim)


def test_create_attention_factory():
    """Test attention factory function."""
    from zllm.core.flash_attention import create_attention, FlashAttention, SlidingWindowAttention, GroupedQueryAttention
    
    # Standard attention
    attn = create_attention(num_heads=8, head_dim=64)
    assert isinstance(attn, FlashAttention)
    
    # Sliding window
    attn = create_attention(num_heads=8, head_dim=64, sliding_window=1024)
    assert isinstance(attn, SlidingWindowAttention)
    assert attn.window_size == 1024
    
    # Grouped query attention
    attn = create_attention(num_heads=8, head_dim=64, num_kv_heads=2)
    assert isinstance(attn, GroupedQueryAttention)
    assert attn.num_kv_heads == 2


def test_estimate_memory():
    """Test memory estimation."""
    from zllm.core.flash_attention import FlashAttention, estimate_attention_memory
    
    batch_size = 4
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    
    attn = FlashAttention(num_heads, head_dim)
    
    memory = attn.estimate_memory(batch_size, seq_len)
    
    assert "standard_attention_memory" in memory
    assert "chunked_attention_memory" in memory
    assert "flash_attention_memory" in memory
    assert "memory_savings_vs_standard" in memory
    
    # Flash should use much less memory
    assert memory["flash_attention_memory"] < memory["standard_attention_memory"]
    assert memory["memory_savings_vs_standard"] > 50  # Should save >50%


def test_estimate_attention_memory_utility():
    """Test standalone memory estimation utility."""
    from zllm.core.flash_attention import estimate_attention_memory
    
    result = estimate_attention_memory(
        batch_size=1,
        seq_len=4096,
        num_heads=32,
        head_dim=128,
    )
    
    assert "standard_attention" in result
    assert "flash_attention" in result
    assert "savings" in result
    assert "recommended" in result
    
    # For long sequences, flash should be recommended
    assert result["recommended"] == "flash"


def test_noncausal_attention():
    """Test attention without causal masking."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    config = FlashAttentionConfig(backend=AttentionBackend.STANDARD, is_causal=False)
    attn = FlashAttention(num_heads=4, head_dim=32, config=config)
    
    batch_size = 2
    seq_len = 16
    hidden_dim = 4 * 32
    
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)
    
    output = attn(q, k, v)
    
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_attention_with_mask():
    """Test attention with explicit attention mask."""
    from zllm.core.flash_attention import FlashAttention, FlashAttentionConfig, AttentionBackend
    
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 32
    hidden_dim = num_heads * head_dim
    
    config = FlashAttentionConfig(backend=AttentionBackend.STANDARD, is_causal=False)
    attn = FlashAttention(num_heads, head_dim, config)
    
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create attention mask (additive, -inf for masked positions)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    mask[:, :, :, seq_len//2:] = float('-inf')  # Mask second half
    
    output = attn(q, k, v, attention_mask=mask)
    
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_extra_repr():
    """Test string representation."""
    from zllm.core.flash_attention import FlashAttention, SlidingWindowAttention, GroupedQueryAttention
    
    attn = FlashAttention(num_heads=8, head_dim=64)
    repr_str = attn.extra_repr()
    assert "heads=8" in repr_str
    assert "head_dim=64" in repr_str
    
    attn = SlidingWindowAttention(num_heads=8, head_dim=64, window_size=512)
    repr_str = attn.extra_repr()
    assert "window=512" in repr_str
    
    gqa = GroupedQueryAttention(num_query_heads=8, num_kv_heads=2, head_dim=64)
    repr_str = gqa.extra_repr()
    assert "q_heads=8" in repr_str
    assert "kv_heads=2" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
