"""
zllm - Memory-efficient LLM inference engine for everyone.

Run large language models on limited hardware with zero configuration.
"""

__version__ = "0.1.0"
__author__ = "zllm Team"

from zllm.core.engine import ZLLM
from zllm.core.config import ZLLMConfig

__all__ = ["ZLLM", "ZLLMConfig", "__version__"]
