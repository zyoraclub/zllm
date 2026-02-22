"""
Tests for the ZLLM engine.
"""

import pytest
from zllm import ZLLM, ZLLMConfig


class TestZLLMConfig:
    """Tests for ZLLMConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ZLLMConfig()
        
        assert config.device == "auto"
        assert config.enable_cache is True
        assert config.enable_semantic_cache is True
        assert config.temperature == 0.7
        assert config.max_new_tokens == 2048
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ZLLMConfig(
            device="cpu",
            temperature=0.5,
            quantization="int4",
        )
        
        assert config.device == "cpu"
        assert config.temperature == 0.5
        assert config.quantization == "int4"


class TestHardwareDetection:
    """Tests for hardware detection."""
    
    def test_detect_hardware(self):
        """Test hardware detection runs without error."""
        from zllm.hardware.auto_detect import detect_hardware
        
        info = detect_hardware()
        
        assert info is not None
        assert info.system is not None
        assert info.system.total_ram > 0
        assert info.system.cpu_count > 0


class TestMemoryCache:
    """Tests for memory cache."""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic cache operations."""
        from zllm.cache.memory import MemoryCache
        
        cache = MemoryCache(max_size=10)
        
        await cache.set(
            key="test1",
            prompt="Hello",
            response="World",
            model_id="test-model",
        )
        
        entry = await cache.get("test1")
        
        assert entry is not None
        assert entry.prompt == "Hello"
        assert entry.response == "World"
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from zllm.cache.memory import MemoryCache
        
        cache = MemoryCache(max_size=2)
        
        await cache.set("k1", "p1", "r1", "model")
        await cache.set("k2", "p2", "r2", "model")
        await cache.set("k3", "p3", "r3", "model")
        
        # k1 should be evicted
        assert await cache.get("k1") is None
        assert await cache.get("k2") is not None
        assert await cache.get("k3") is not None


class TestQuantization:
    """Tests for quantization config."""
    
    def test_int4_config(self):
        """Test INT4 quantization config."""
        from zllm.quantization.base import QuantizationConfig
        
        config = QuantizationConfig.int4()
        
        assert config.bits == 4
    
    def test_auto_quantizer(self):
        """Test auto quantizer recommendations."""
        from zllm.quantization.base import AutoQuantizer, QuantizationType
        
        # Should recommend no quantization if plenty of memory
        config = AutoQuantizer.get_recommended_config(
            model_size_gb=10,
            available_memory_gb=50,
        )
        assert config.quant_type == QuantizationType.NONE
        
        # Should recommend INT4 for large model with limited memory
        config = AutoQuantizer.get_recommended_config(
            model_size_gb=70,
            available_memory_gb=16,
        )
        assert config.quant_type == QuantizationType.INT4


class TestBackendDetection:
    """Tests for backend detection and loading."""
    
    def test_detect_backend_standard(self):
        """Test detection of standard models."""
        from zllm.models.loader import ModelLoader
        
        loader = ModelLoader()
        
        # Standard model should use bitsandbytes
        assert loader.detect_backend("meta-llama/Llama-2-7b-chat-hf") == "bitsandbytes"
        assert loader.detect_backend("Qwen/Qwen2-7B-Instruct") == "bitsandbytes"
    
    def test_detect_backend_awq(self):
        """Test detection of AWQ models."""
        from zllm.models.loader import ModelLoader, AWQ_AVAILABLE
        
        loader = ModelLoader()
        
        if AWQ_AVAILABLE:
            assert loader.detect_backend("TheBloke/Llama-2-7B-Chat-AWQ") == "awq"
            assert loader.detect_backend("Qwen2-7B-Instruct-AWQ") == "awq"
        else:
            # Should raise ImportError for AWQ models without library
            try:
                loader.detect_backend("TheBloke/Llama-2-7B-Chat-AWQ")
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "auto-awq" in str(e)
    
    def test_detect_backend_gptq(self):
        """Test detection of GPTQ models."""
        from zllm.models.loader import ModelLoader, GPTQ_AVAILABLE
        
        loader = ModelLoader()
        
        if GPTQ_AVAILABLE:
            assert loader.detect_backend("TheBloke/Llama-2-7B-Chat-GPTQ") == "gptq"
        else:
            try:
                loader.detect_backend("TheBloke/Llama-2-7B-Chat-GPTQ")
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "auto-gptq" in str(e)
    
    def test_detect_backend_gguf(self):
        """Test detection of GGUF files."""
        from zllm.models.loader import ModelLoader, GGUF_AVAILABLE
        
        loader = ModelLoader()
        
        if GGUF_AVAILABLE:
            assert loader.detect_backend("model.gguf") == "gguf"
            assert loader.detect_backend("/path/to/llama-2-7b-q4_0.gguf") == "gguf"
        else:
            try:
                loader.detect_backend("model.gguf")
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "llama-cpp-python" in str(e)
    
    def test_get_available_backends(self):
        """Test backend availability reporting."""
        from zllm.models.loader import ModelLoader
        
        loader = ModelLoader()
        backends = loader.get_available_backends()
        
        assert "bitsandbytes" in backends
        assert "awq" in backends
        assert "gptq" in backends
        assert "gguf" in backends
        # bitsandbytes is always available
        assert backends["bitsandbytes"] is True
    
    def test_config_backend_field(self):
        """Test ZLLMConfig backend field."""
        from zllm import ZLLMConfig
        
        # Default should be bitsandbytes
        config = ZLLMConfig()
        assert config.backend == "bitsandbytes"
        
        # Should accept other backends
        config = ZLLMConfig(backend="awq")
        assert config.backend == "awq"
        
        config = ZLLMConfig(backend="gguf")
        assert config.backend == "gguf"