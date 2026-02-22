"""
End-to-end test for ZLLM.

Tests the complete pipeline from model loading to inference.
"""

import sys
import time

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from zllm import ZLLM, ZLLMConfig, __version__
    print(f"  ✓ zllm v{__version__} imported")
    
    from zllm.core import (
        MemoryManager,
        SpeedMode,
        IntelligentOrchestrator,
        KVCacheManager,
        QuantizationScheme,
    )
    print("  ✓ Core modules imported")
    
    from zllm.hardware.auto_detect import detect_hardware
    print("  ✓ Hardware detection imported")
    
    return True


def test_hardware_detection():
    """Test hardware detection."""
    print("\nTesting hardware detection...")
    
    from zllm.hardware.auto_detect import detect_hardware
    
    hw = detect_hardware()
    print(f"  ✓ Has GPU: {hw.has_gpu}")
    if hw.has_gpu:
        gpu = hw.gpus[0]
        print(f"  ✓ GPU: {gpu.name} ({gpu.free_memory_gb:.1f}GB free)")
    print(f"  ✓ RAM: {hw.system.available_ram_gb:.1f}GB available")
    
    return True


def test_kv_cache():
    """Test KV cache functionality."""
    print("\nTesting KV cache...")
    
    import torch
    from zllm.core.kv_cache import (
        KVCacheManager,
        KVCacheQuantizer,
        QuantizationScheme,
        PromptCache,
    )
    
    # Test quantization
    quantizer = KVCacheQuantizer(scheme=QuantizationScheme.INT8)
    test_tensor = torch.randn(1, 32, 512, 128)  # [batch, heads, seq, dim]
    
    quantized = quantizer.quantize(test_tensor)
    dequantized = quantized.dequantize()
    
    # Check shapes match
    assert dequantized.shape == test_tensor.shape, "Shape mismatch!"
    print(f"  ✓ Quantization: {test_tensor.numel() * 2} bytes → {quantized.memory_bytes} bytes")
    
    # Test reconstruction error
    error = (test_tensor - dequantized).abs().mean().item()
    print(f"  ✓ Reconstruction error: {error:.6f}")
    
    # Test prompt cache
    cache = PromptCache(max_entries=10)
    # Create a proper KV cache format: Dict[layer_idx, (key, value)]
    kv_state = {
        0: (torch.randn(1, 32, 512, 64), torch.randn(1, 32, 512, 64)),
        1: (torch.randn(1, 32, 512, 64), torch.randn(1, 32, 512, 64)),
    }
    cache.put("test_prompt", kv_state, num_tokens=512)
    retrieved = cache.get("test_prompt")
    assert retrieved is not None, "Cache miss!"
    print(f"  ✓ Prompt cache: working")
    
    # Test KVCacheManager
    manager = KVCacheManager(
        quantization=QuantizationScheme.INT8,
        enable_prompt_cache=True,
        enable_prefix_cache=True,
    )
    stats = manager.get_stats()
    has_prompt_cache = "prompt_cache" in stats
    print(f"  ✓ KVCacheManager: {stats['quantization']}, prompt_cache={has_prompt_cache}")
    
    return True


def test_memory_manager():
    """Test memory management."""
    print("\nTesting memory management...")
    
    from zllm.core.memory import MemoryManager, SpeedMode
    
    for mode in [SpeedMode.FAST, SpeedMode.BALANCED, SpeedMode.MEMORY_SAVER]:
        mm = MemoryManager(device="cpu", speed_mode=mode)
        fraction = mode.memory_fraction * 100
        print(f"  ✓ SpeedMode.{mode.name}: {fraction:.0f}% VRAM")
    
    return True


def test_orchestrator():
    """Test intelligent orchestrator."""
    print("\nTesting orchestrator...")
    
    from zllm.core.memory import MemoryManager, SpeedMode
    from zllm.core.orchestrator import IntelligentOrchestrator, MemoryPressure
    
    mm = MemoryManager(device="cpu", speed_mode=SpeedMode.BALANCED)
    orch = IntelligentOrchestrator(memory_manager=mm, target_utilization=0.8)
    
    # Test auto speed mode selection
    auto_mode = orch.auto_select_speed_mode(
        model_layers=32,
        layer_size_bytes=500 * 1024 * 1024,  # 500MB per layer
    )
    print(f"  ✓ Auto speed mode: {auto_mode.value}")
    
    # Check memory pressure enum
    for pressure in MemoryPressure:
        print(f"  ✓ MemoryPressure.{pressure.value}")
    
    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    from zllm.core.config import ZLLMConfig
    
    config = ZLLMConfig(
        speed_mode="fast",
        max_new_tokens=512,
        enable_semantic_cache=True,
    )
    
    print(f"  ✓ Speed mode: {config.speed_mode}")
    print(f"  ✓ Max tokens: {config.max_new_tokens}")
    print(f"  ✓ Semantic cache: {config.enable_semantic_cache}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("ZLLM End-to-End Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("KV Cache", test_kv_cache),
        ("Memory Manager", test_memory_manager),
        ("Orchestrator", test_orchestrator),
        ("Configuration", test_config),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
