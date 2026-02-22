"""
ZLLM Engine - Reliable LLM inference for GGUF models.

Supports:
- GGUF format parsing
- Quantized inference (Q4_K_M, Q5_K, Q6_K, Q8_0, etc.)
- llama.cpp backend for production reliability
- Custom CUDA/Triton kernels (optional)
- Memory-efficient layer streaming
"""

from .gguf_parser import GGUFParser, GGUFMetadata, GGUFTensor, inspect_gguf
from .llama_cpp_backend import LlamaCppBackend, GenerationConfig, LLAMA_CPP_AVAILABLE
from .quantization import (
    QuantType,
    dequantize_q4_0,
    dequantize_q4_k,
    dequantize_q8_0,
    dequantize_tensor,
    get_quantization_info,
)
from .inference import ZLLMInferenceEngine, InferenceConfig, load_engine
from .tokenizer import SimpleTokenizer, ChatTemplate, load_tokenizer_from_gguf
from .cuda_kernels import (
    dequant_q8_0_cuda,
    dequant_q4_0_cuda,
    rms_norm_cuda,
    apply_rope_cuda,
    flash_attention_cuda,
    is_triton_available,
    get_backend_info,
)

__all__ = [
    # Parser
    "GGUFParser",
    "GGUFMetadata",
    "GGUFTensor",
    "inspect_gguf",
    # Quantization
    "QuantType",
    "dequantize_q4_0",
    "dequantize_q4_k",
    "dequantize_q8_0",
    "dequantize_tensor",
    "get_quantization_info",
    # Inference
    "ZLLMInferenceEngine",
    "InferenceConfig",
    "load_engine",
    # Tokenizer
    "SimpleTokenizer",
    "ChatTemplate",
    "load_tokenizer_from_gguf",
    # CUDA Kernels
    "dequant_q8_0_cuda",
    "dequant_q4_0_cuda",
    "rms_norm_cuda",
    "apply_rope_cuda",
    "flash_attention_cuda",
    "is_triton_available",
    "get_backend_info",
]
