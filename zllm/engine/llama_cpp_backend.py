"""
LLAMA.CPP Backend - Production-ready inference using llama-cpp-python.

This provides reliable, battle-tested inference for GGUF models.
"""

from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass
import os

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None


class LlamaCppBackend:
    """
    Production inference backend using llama-cpp-python.
    
    Provides:
    - Reliable GGUF inference (Q4_K, Q5_K, Q6_K, etc.)
    - Streaming generation
    - OpenAI-compatible output format
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the llama.cpp backend.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            n_threads: Number of CPU threads (None = auto)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            verbose: Print llama.cpp logs
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        
        # Initialize llama.cpp
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads or os.cpu_count(),
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        
        # Extract model info
        self.vocab_size = self.llm.n_vocab()
        self.n_embd = self.llm.n_embd()
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text completion.
        
        Args:
            prompt: Input text
            config: Generation configuration
            stream: Whether to stream tokens
            
        Returns:
            OpenAI-compatible completion response
        """
        config = config or GenerationConfig()
        
        if stream:
            return self._generate_stream(prompt, config)
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop,
            echo=False,
        )
        
        return response
    
    def _generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> Iterator[Dict[str, Any]]:
        """Stream tokens as they're generated."""
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop,
            echo=False,
            stream=True,
        ):
            yield chunk
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Chat completion (if model supports chat format).
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            config: Generation configuration
            stream: Whether to stream tokens
            
        Returns:
            OpenAI-compatible chat completion response
        """
        config = config or GenerationConfig()
        
        if stream:
            return self._chat_stream(messages, config)
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop,
        )
        
        return response
    
    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Iterator[Dict[str, Any]]:
        """Stream chat tokens."""
        for chunk in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop,
            stream=True,
        ):
            yield chunk
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        return self.llm.tokenize(text.encode('utf-8'))
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.llm.detokenize(tokens).decode('utf-8')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_path": self.model_path,
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_ctx": self.n_ctx,
            "backend": "llama.cpp",
        }
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
