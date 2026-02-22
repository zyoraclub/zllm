"""
ZLLM Native Engine - Our own inference engine.

No external dependencies - 100% controlled by us.

Supports:
- GGUF format parsing
- Quantized inference (Q4_K_M, Q8_0, etc.)
- Custom CUDA kernels
- Memory-efficient layer streaming
"""

from .gguf_parser import GGUFParser, GGUFMetadata, GGUFTensor, inspect_gguf
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
]
