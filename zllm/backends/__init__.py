"""
Backend implementations for different model formats.

Available backends:
- gguf: GGUF format via llama-cpp-python (fastest for local)
- bitsandbytes: HuggingFace with INT8/INT4 quantization (default)
"""

from .gguf import (
    GGUF_AVAILABLE,
    GGUFModel,
    GGUFModelInfo,
    load_gguf,
    download_gguf,
    list_gguf_files,
    recommend_gguf_file,
)

__all__ = [
    "GGUF_AVAILABLE",
    "GGUFModel",
    "GGUFModelInfo",
    "load_gguf",
    "download_gguf",
    "list_gguf_files",
    "recommend_gguf_file",
]
