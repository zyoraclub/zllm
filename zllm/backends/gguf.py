"""
GGUF backend for zllm using llama-cpp-python.

GGUF is the standard format for local LLM inference (Ollama, LM Studio, etc.)
Provides 3-4x faster inference than bitsandbytes.
"""

import os
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List
from dataclasses import dataclass

try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

from huggingface_hub import hf_hub_download, list_repo_files, HfApi


@dataclass
class GGUFModelInfo:
    """Information about a GGUF model."""
    path: Path
    filename: str
    size_gb: float
    quantization: str  # e.g., Q4_K_M, Q8_0
    
    @classmethod
    def from_filename(cls, path: Path) -> "GGUFModelInfo":
        """Extract info from GGUF filename."""
        filename = path.name
        size_gb = path.stat().st_size / (1024 ** 3)
        
        # Extract quantization from filename
        # Common patterns: q4_k_m, q8_0, q5_k_s, etc.
        quant = "unknown"
        lower = filename.lower()
        for q in ["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_k_s", 
                  "q4_k_m", "q5_0", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "fp16"]:
            if q in lower:
                quant = q.upper()
                break
        
        return cls(path=path, filename=filename, size_gb=size_gb, quantization=quant)


def list_gguf_files(repo_id: str) -> List[str]:
    """List all GGUF files in a HuggingFace repo."""
    try:
        files = list_repo_files(repo_id)
        return [f for f in files if f.endswith('.gguf')]
    except Exception:
        return []


def recommend_gguf_file(repo_id: str, max_size_gb: float = 8.0) -> Optional[str]:
    """
    Recommend the best GGUF file based on available memory.
    
    Priority: Q4_K_M (best balance) > Q5_K_M > Q8_0 > others
    """
    files = list_gguf_files(repo_id)
    if not files:
        return None
    
    # Preference order for quantization
    preference = ['q4_k_m', 'q4_k_s', 'q5_k_m', 'q5_k_s', 'q3_k_m', 'q8_0', 'q6_k']
    
    for pref in preference:
        for f in files:
            if pref in f.lower():
                return f
    
    # Fallback to first GGUF file
    return files[0] if files else None


def download_gguf(
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Download a GGUF file from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo (e.g., "Qwen/Qwen2-7B-Instruct-GGUF")
        filename: Specific GGUF file, or None to auto-select best
        cache_dir: Cache directory for downloads
    
    Returns:
        Path to downloaded GGUF file
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "zllm" / "gguf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # If filename not specified, find the best one
    if filename is None:
        filename = recommend_gguf_file(repo_id)
        if filename is None:
            raise ValueError(f"No GGUF files found in {repo_id}")
    
    # Download
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )
    
    return Path(path)


class GGUFModel:
    """
    Wrapper around llama-cpp-python's Llama for zllm integration.
    
    Provides a consistent interface matching HuggingFace models.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Load a GGUF model.
        
        Args:
            model_path: Path to .gguf file
            n_ctx: Context window size
            n_gpu_layers: GPU layers (-1 = all, 0 = CPU only)
            n_threads: CPU threads (None = auto)
            verbose: Enable verbose logging
        """
        if not GGUF_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models.\n"
                "Install with: pip install llama-cpp-python\n"
                "For GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )
        
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        
        # Auto-detect threads
        if n_threads is None:
            n_threads = min(os.cpu_count() or 4, 8)
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )
        
        # Get model info
        self.info = GGUFModelInfo.from_filename(self.model_path)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: Stop sequences
            stream: Whether to stream output
        
        Returns:
            Generated text
        """
        if stream:
            return self.generate_stream(
                prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop
            )
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
        )
        
        return output["choices"][0]["text"]
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Stream generated tokens."""
        for output in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            stream=True,
        ):
            yield output["choices"][0]["text"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Chat with the model using messages format.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream output
        
        Returns:
            Assistant's response
        """
        if stream:
            return self.chat_stream(messages, max_tokens, temperature)
        
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return output["choices"][0]["message"]["content"]
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream chat response."""
        for output in self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            delta = output["choices"][0].get("delta", {})
            if "content" in delta:
                yield delta["content"]
    
    def __repr__(self) -> str:
        return f"GGUFModel({self.info.filename}, {self.info.quantization}, {self.info.size_gb:.1f}GB)"


def load_gguf(
    model_id: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    cache_dir: Optional[Path] = None,
) -> GGUFModel:
    """
    Load a GGUF model from local path or HuggingFace.
    
    Args:
        model_id: Local .gguf path OR HuggingFace repo (e.g., "Qwen/Qwen2-7B-Instruct-GGUF")
        n_ctx: Context window size
        n_gpu_layers: GPU layers (-1 = all)
        cache_dir: Cache directory
    
    Returns:
        GGUFModel ready for inference
    
    Examples:
        # Local file
        model = load_gguf("/path/to/model.gguf")
        
        # HuggingFace (auto-downloads best quantization)
        model = load_gguf("Qwen/Qwen2-7B-Instruct-GGUF")
        
        # HuggingFace with specific file
        model = load_gguf("Qwen/Qwen2-7B-Instruct-GGUF/qwen2-7b-instruct-q4_k_m.gguf")
    """
    path = Path(model_id)
    
    # Check if it's a local file
    if path.exists() and path.suffix == '.gguf':
        return GGUFModel(str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    
    # It's a HuggingFace repo
    if '/' in model_id:
        # Check if specific file is specified
        if model_id.endswith('.gguf'):
            # Format: "repo/owner/filename.gguf"
            parts = model_id.rsplit('/', 1)
            repo_id = parts[0]
            filename = parts[1]
        else:
            repo_id = model_id
            filename = None
        
        # Download from HuggingFace
        local_path = download_gguf(repo_id, filename, cache_dir)
        return GGUFModel(str(local_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    
    raise ValueError(
        f"Cannot find GGUF model: {model_id}\n"
        "Provide either:\n"
        "  - Local path: /path/to/model.gguf\n"
        "  - HuggingFace repo: Qwen/Qwen2-7B-Instruct-GGUF"
    )
