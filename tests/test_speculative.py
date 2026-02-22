"""Tests for Speculative Decoding."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Test imports
def test_speculative_imports():
    """Test all speculative decoding components import correctly."""
    from zllm.core.speculative import (
        SpeculativeDecoder,
        SpeculativeDecoderWithCache,
        SpeculativeConfig,
        SpeculativeStats,
        AcceptanceMethod,
        create_speculative_pair,
        benchmark_speculative,
    )
    
    # Also test from main module
    from zllm.core import (
        SpeculativeDecoder,
        SpeculativeConfig,
        AcceptanceMethod,
    )


def test_speculative_config():
    """Test SpeculativeConfig defaults and customization."""
    from zllm.core.speculative import SpeculativeConfig, AcceptanceMethod
    
    # Default config
    config = SpeculativeConfig()
    assert config.num_draft_tokens == 5
    assert config.acceptance_method == AcceptanceMethod.SAMPLING
    assert config.acceptance_threshold == 0.1
    assert config.min_acceptance_rate == 0.3
    
    # Custom config
    config = SpeculativeConfig(
        draft_model_id="test/draft",
        num_draft_tokens=8,
        acceptance_method=AcceptanceMethod.GREEDY,
    )
    assert config.draft_model_id == "test/draft"
    assert config.num_draft_tokens == 8
    assert config.acceptance_method == AcceptanceMethod.GREEDY


def test_speculative_stats():
    """Test SpeculativeStats tracking and calculations."""
    from zllm.core.speculative import SpeculativeStats
    
    stats = SpeculativeStats()
    assert stats.acceptance_rate == 0.0
    assert stats.speedup_factor == 1.0
    
    # Simulate some decoding
    stats.total_draft_tokens = 20
    stats.accepted_tokens = 15
    stats.rejected_tokens = 5
    stats.target_forward_passes = 5
    stats.draft_forward_passes = 20
    
    # Check calculations
    assert stats.acceptance_rate == 0.75  # 15/20
    # Speedup: (15 + 5) / 5 = 4x
    assert stats.speedup_factor == 4.0
    
    # Test string representation
    stats_str = str(stats)
    assert "acceptance_rate=75.0%" in stats_str
    assert "speedup=4.00x" in stats_str


def test_acceptance_methods():
    """Test different acceptance methods."""
    from zllm.core.speculative import AcceptanceMethod
    
    assert AcceptanceMethod.GREEDY.value == "greedy"
    assert AcceptanceMethod.SAMPLING.value == "sampling"
    assert AcceptanceMethod.THRESHOLD.value == "threshold"


@dataclass
class MockModelOutput:
    """Mock model output."""
    logits: torch.Tensor
    past_key_values: tuple = None


def create_mock_model(vocab_size: int = 1000):
    """Create a mock model for testing."""
    model = MagicMock()
    
    def forward(input_ids, past_key_values=None, use_cache=False):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Random logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Make a specific token most likely
        logits[:, :, 42] = 10.0  # Token 42 is always most likely
        
        return MockModelOutput(logits=logits)
    
    model.side_effect = forward
    model.return_value = forward(torch.zeros(1, 1, dtype=torch.long))
    model.eval = MagicMock()
    model.__call__ = forward
    
    return model


def create_mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.encode = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    tokenizer.decode = MagicMock(return_value="test")
    return tokenizer


def test_speculative_decoder_init():
    """Test SpeculativeDecoder initialization."""
    from zllm.core.speculative import SpeculativeDecoder, SpeculativeConfig
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        device="cpu",
    )
    
    assert decoder.target_model == target
    assert decoder.draft_model == draft
    assert decoder.config.num_draft_tokens == 5
    assert decoder.stats.accepted_tokens == 0


def test_speculative_decoder_reset():
    """Test stats reset."""
    from zllm.core.speculative import SpeculativeDecoder
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        device="cpu",
    )
    
    # Simulate some activity
    decoder.stats.accepted_tokens = 100
    decoder.stats.rejected_tokens = 20
    decoder._fallback_mode = True
    
    # Reset
    decoder.reset_stats()
    
    assert decoder.stats.accepted_tokens == 0
    assert decoder.stats.rejected_tokens == 0
    assert decoder._fallback_mode is False


def test_sampling_function():
    """Test token sampling with various parameters."""
    from zllm.core.speculative import SpeculativeDecoder, SpeculativeConfig
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        device="cpu",
    )
    
    # Test greedy sampling (temperature=0)
    logits = torch.tensor([1.0, 2.0, 10.0, 3.0, 1.0])  # Index 2 is max
    token = decoder._sample(logits, temperature=0, top_p=1.0, top_k=0)
    assert token.item() == 2
    
    # Test with top_k
    logits = torch.tensor([1.0, 2.0, 10.0, 3.0, 1.0])
    for _ in range(10):
        token = decoder._sample(logits, temperature=1.0, top_p=1.0, top_k=2)
        assert token.item() in [2, 3]  # Only top 2 tokens allowed


def test_get_stats():
    """Test get_stats returns correct dictionary."""
    from zllm.core.speculative import SpeculativeDecoder
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        device="cpu",
    )
    
    decoder.stats.accepted_tokens = 10
    decoder.stats.total_draft_tokens = 15
    
    stats = decoder.get_stats()
    
    assert "acceptance_rate" in stats
    assert "speedup_factor" in stats
    assert "accepted_tokens" in stats
    assert "is_fallback_mode" in stats
    assert stats["accepted_tokens"] == 10


def test_auto_select_draft():
    """Test automatic draft model selection."""
    from zllm.core.speculative import _auto_select_draft
    
    # Known model
    draft = _auto_select_draft("meta-llama/Llama-2-70b-chat-hf")
    assert draft == "meta-llama/Llama-2-7b-chat-hf"
    
    # 70B -> 7B conversion
    draft = _auto_select_draft("org/some-model-70b")
    assert "7b" in draft.lower()
    
    # 13B -> 7B conversion
    draft = _auto_select_draft("org/model-13B-instruct")
    assert "7B" in draft


def test_speculative_decoder_with_cache():
    """Test cache-enabled speculative decoder."""
    from zllm.core.speculative import SpeculativeDecoderWithCache
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    decoder = SpeculativeDecoderWithCache(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        device="cpu",
    )
    
    assert decoder._target_kv_cache is None
    assert decoder._draft_kv_cache is None
    
    # Test cache reset
    decoder._target_kv_cache = "some_cache"
    decoder.reset_cache()
    assert decoder._target_kv_cache is None


def test_acceptance_greedy():
    """Test greedy acceptance method."""
    from zllm.core.speculative import SpeculativeDecoder, SpeculativeConfig, AcceptanceMethod
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    config = SpeculativeConfig(acceptance_method=AcceptanceMethod.GREEDY)
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        config=config,
        device="cpu",
    )
    
    # Create target logits where token 5 is the argmax
    target_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 1.0])
    
    # Token 5 should be accepted (matches argmax)
    accepted = decoder._accept_token(
        draft_token=5,
        draft_prob=0.5,
        target_logits=target_logits,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    assert accepted is True
    
    # Token 0 should be rejected (doesn't match argmax)
    accepted = decoder._accept_token(
        draft_token=0,
        draft_prob=0.5,
        target_logits=target_logits,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    assert accepted is False


def test_acceptance_threshold():
    """Test threshold acceptance method."""
    from zllm.core.speculative import SpeculativeDecoder, SpeculativeConfig, AcceptanceMethod
    
    target = create_mock_model()
    draft = create_mock_model()
    tokenizer = create_mock_tokenizer()
    
    config = SpeculativeConfig(
        acceptance_method=AcceptanceMethod.THRESHOLD,
        acceptance_threshold=0.3,
    )
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        config=config,
        device="cpu",
    )
    
    # Create logits where softmax gives high prob to token 0
    # With high logit, softmax will give high probability
    target_logits = torch.tensor([10.0, 1.0, 1.0, 1.0])  # Token 0 has ~90% prob
    
    # Token 0 should be accepted (high probability)
    accepted = decoder._accept_token(
        draft_token=0,
        draft_prob=0.5,
        target_logits=target_logits,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    assert accepted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
