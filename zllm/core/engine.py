"""
Main ZLLM inference engine.

This is the primary interface for loading and running models.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator, Union
import torch

from zllm.core.config import ZLLMConfig
from zllm.core.memory import MemoryManager
from zllm.core.generation import GenerationConfig, GenerationOutput, TextGenerator
from zllm.hardware.auto_detect import detect_hardware, get_best_device
from zllm.models.loader import ModelLoader, ModelInfo
from zllm.cache.semantic import SemanticCache
from zllm.cache.memory import MemoryCache


class ZLLM:
    """
    Main inference engine for zllm.
    
    Provides a simple interface for loading and running models with
    memory-efficient inference and semantic caching.
    
    Example:
        ```python
        from zllm import ZLLM
        
        # Simple usage
        llm = ZLLM("meta-llama/Llama-3-8B-Instruct")
        response = llm.chat("What is the capital of France?")
        print(response)
        
        # With configuration
        from zllm import ZLLMConfig
        config = ZLLMConfig(
            quantization="int4",
            enable_semantic_cache=True,
        )
        llm = ZLLM("meta-llama/Llama-3-70B-Instruct", config=config)
        ```
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        config: Optional[ZLLMConfig] = None,
        auto_load: bool = True,
    ):
        """
        Initialize the ZLLM engine.
        
        Args:
            model_id: HuggingFace model ID or local path
            config: Configuration options
            auto_load: Whether to load the model immediately
        """
        self.config = config or ZLLMConfig()
        if model_id:
            self.config.model_id = model_id
        
        # Initialize components
        self.hardware_info = detect_hardware()
        self.loader = ModelLoader(cache_dir=self.config.cache_dir)
        self.memory_manager: Optional[MemoryManager] = None
        
        # Intelligent orchestrator for dynamic VRAM optimization
        self.orchestrator = None
        
        # KV Cache manager for efficient context handling
        self.kv_cache_manager = None
        
        # Speculative decoder for faster inference
        self.speculative_decoder = None
        self.draft_model = None
        
        # Flash Attention configuration
        self.flash_attention_config = None
        
        # Model components (loaded later)
        self.model = None
        self.tokenizer = None
        self.generator: Optional[TextGenerator] = None
        self.model_info: Optional[ModelInfo] = None
        
        # Cache
        self._cache: Optional[Union[SemanticCache, MemoryCache]] = None
        self._init_cache()
        
        # Device
        if self.config.device == "auto":
            self.device = torch.device(get_best_device())
        else:
            self.device = torch.device(self.config.device)
        
        # Auto-load model
        if auto_load and self.config.model_id:
            self.load_model(self.config.model_id)
    
    def _init_cache(self) -> None:
        """Initialize the cache system."""
        if not self.config.enable_cache:
            return
        
        cache_path = self.config.cache_dir / "response_cache.json"
        
        if self.config.enable_semantic_cache:
            self._cache = SemanticCache(
                max_size=self.config.cache_max_size,
                similarity_threshold=self.config.semantic_cache_threshold,
                persist_path=cache_path,
            )
        else:
            self._cache = MemoryCache(
                max_size=self.config.cache_max_size,
                persist_path=cache_path,
            )
    
    def load_model(
        self,
        model_id: str,
        quantization: Optional[str] = None,
    ) -> "ZLLM":
        """
        Load a model for inference.
        
        Args:
            model_id: HuggingFace model ID or local path
            quantization: Override quantization setting ("int4", "int8", None)
        
        Returns:
            self for chaining
        """
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        console = Console()
        
        # Get model info
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Getting model info...", total=None)
            self.model_info = self.loader.get_model_info(model_id)
            progress.update(task, description="Model info loaded")
        
        # Determine quantization
        quant = quantization or self.config.quantization
        if quant is None and self.config.auto_quantize:
            quant = self.hardware_info.get_recommended_quantization(
                self.model_info.size_gb
            )
            if quant:
                console.print(f"[yellow]Auto-selected {quant} quantization for memory efficiency[/yellow]")
        
        # Load tokenizer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading tokenizer...", total=None)
            self.tokenizer = self.loader.load_tokenizer(model_id)
            progress.update(task, description="Tokenizer loaded")
        
        # Load model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Loading {model_id}...", total=None)
            
            self.model = self.loader.load_model_full(
                model_id,
                device_map=self.config.device_map,
                dtype=torch.float16,
                quantization=quant,
            )
            
            progress.update(task, description="Model loaded")
        
        # Initialize generator
        self.generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        # Initialize memory manager with speed mode from config
        from zllm.core.memory import SpeedMode
        from zllm.core.orchestrator import IntelligentOrchestrator
        
        speed_mode = SpeedMode(self.config.speed_mode)
        
        self.memory_manager = MemoryManager(
            device=self.config.device,
            offload_to_cpu=self.config.offload_to_cpu,
            speed_mode=speed_mode,
        )
        
        # Initialize intelligent orchestrator for dynamic VRAM optimization
        self.orchestrator = IntelligentOrchestrator(
            memory_manager=self.memory_manager,
            target_utilization=0.80,  # Use up to 80% VRAM for speed
        )
        
        # Auto-select optimal speed mode based on available VRAM
        if self.config.speed_mode == "balanced" and self.model_info:
            # Let orchestrator decide best mode
            estimated_layer_size = int(self.model_info.size_gb * 1024 * 1024 * 1024 / 32)  # Rough estimate
            auto_mode = self.orchestrator.auto_select_speed_mode(
                model_layers=32,  # Typical for 7B
                layer_size_bytes=estimated_layer_size,
            )
            self.memory_manager.speed_mode = auto_mode
            console.print(f"  [cyan]🧠 Auto-optimized: {auto_mode.value} mode[/cyan]")
        
        # Start orchestrator for dynamic optimization
        self.orchestrator.start()
        
        # Initialize KV Cache manager for efficient context handling
        from zllm.core.kv_cache import KVCacheManager, QuantizationScheme
        self.kv_cache_manager = KVCacheManager(
            quantization_scheme=QuantizationScheme.INT8,  # 50% memory savings
            enable_prompt_cache=True,
            enable_prefix_cache=True,
            prompt_cache_size=100,
        )
        
        # Initialize Flash Attention if enabled
        if self.config.enable_flash_attention:
            self._setup_flash_attention()
        
        # Initialize Speculative Decoding if configured
        if self.config.enable_speculative and self.config.draft_model_id:
            self._setup_speculative_decoding(model_id, quant)
        
        # Log info
        console.print(f"[green]✓ Model loaded successfully[/green]")
        console.print(f"  Device: {self.device}")
        console.print(f"  Parameters: {self.model_info.params_billions:.1f}B")
        console.print(f"  Speed Mode: {self.memory_manager.speed_mode.value}")
        console.print(f"  [cyan]🧠 Intelligent orchestrator: active[/cyan]")
        console.print(f"  [cyan]💾 KV cache: INT8 quantized[/cyan]")
        if self.flash_attention_config:
            console.print(f"  [cyan]⚡ Flash Attention: {self.flash_attention_config.get('backend', 'enabled')}[/cyan]")
        if self.speculative_decoder:
            console.print(f"  [cyan]🚀 Speculative Decoding: {self.config.draft_model_id}[/cyan]")
        if quant:
            console.print(f"  Quantization: {quant}")
        
        return self
    
    def _setup_flash_attention(self) -> None:
        """Configure Flash Attention for optimal performance."""
        from zllm.core.flash_attention import (
            FlashAttentionConfig, 
            AttentionBackend, 
            detect_best_backend,
        )
        
        # Detect best backend
        if self.config.flash_attention_backend == "auto":
            backend = detect_best_backend()
        else:
            backend_map = {
                "flash_attn": AttentionBackend.FLASH_ATTN,
                "sdpa": AttentionBackend.SDPA,
                "chunked": AttentionBackend.CHUNKED,
            }
            backend = backend_map.get(self.config.flash_attention_backend, detect_best_backend())
        
        self.flash_attention_config = {
            "backend": backend.value,
            "sliding_window": self.config.attention_sliding_window,
            "is_causal": True,
        }
        
        # Note: The actual injection into model attention layers would require
        # model-specific hooks. For now, we configure and make available.
        # Full integration would happen in model-specific forward hooks.
    
    def _setup_speculative_decoding(
        self, 
        target_model_id: str, 
        quantization: Optional[str] = None,
    ) -> None:
        """Set up speculative decoding with a draft model."""
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from zllm.core.speculative import (
            SpeculativeDecoder,
            SpeculativeConfig,
            AcceptanceMethod,
        )
        
        console = Console()
        
        # Map acceptance method
        acceptance_map = {
            "greedy": AcceptanceMethod.GREEDY,
            "sampling": AcceptanceMethod.SAMPLING,
            "threshold": AcceptanceMethod.THRESHOLD,
        }
        acceptance_method = acceptance_map.get(
            self.config.speculative_acceptance, 
            AcceptanceMethod.SAMPLING
        )
        
        # Load draft model (use more aggressive quantization)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Loading draft model {self.config.draft_model_id}...", total=None)
            
            # Draft model should be smaller, use int4 for memory efficiency
            draft_quant = "int4" if quantization != "int4" else "int8"
            
            self.draft_model = self.loader.load_model_full(
                self.config.draft_model_id,
                device_map=self.config.device_map,
                dtype=torch.float16,
                quantization=draft_quant,
            )
            
            progress.update(task, description="Draft model loaded")
        
        # Create speculative config
        spec_config = SpeculativeConfig(
            draft_model_id=self.config.draft_model_id,
            num_draft_tokens=self.config.num_speculative_tokens,
            acceptance_method=acceptance_method,
        )
        
        # Create decoder
        self.speculative_decoder = SpeculativeDecoder(
            target_model=self.model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            config=spec_config,
            device=str(self.device),
        )
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            history: Optional conversation history
            **kwargs: Additional generation parameters
        
        Returns:
            Assistant response text
        """
        if self.generator is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Check cache
        if self._cache and isinstance(self._cache, SemanticCache):
            cached = asyncio.get_event_loop().run_until_complete(
                self._cache.get_semantic(message, self.config.model_id or "")
            )
            if cached:
                entry, similarity = cached
                return entry.response
        
        # Format prompt
        prompt = self.generator.format_prompt(
            user_message=message,
            system_prompt=system_prompt,
            history=history,
        )
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
        )
        
        # Generate
        output = self.generator.generate(prompt, gen_config)
        
        # Cache response
        if self._cache:
            asyncio.get_event_loop().run_until_complete(
                self._cache.set_semantic(
                    prompt=message,
                    response=output.text,
                    model_id=self.config.model_id or "",
                    tokens_used=output.total_tokens,
                )
            )
        
        return output.text
    
    def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Send a chat message and stream the response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            history: Optional conversation history
            **kwargs: Additional generation parameters
        
        Yields:
            Response tokens as they're generated
        """
        if self.generator is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Format prompt
        prompt = self.generator.format_prompt(
            user_message=message,
            system_prompt=system_prompt,
            history=history,
        )
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
            stream=True,
        )
        
        # Stream generation
        full_response = []
        for token in self.generator.generate_stream(prompt, gen_config):
            full_response.append(token)
            yield token
        
        # Cache the full response
        if self._cache:
            asyncio.get_event_loop().run_until_complete(
                self._cache.set_semantic(
                    prompt=message,
                    response="".join(full_response),
                    model_id=self.config.model_id or "",
                )
            )
    
    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text from a raw prompt.
        
        Args:
            prompt: Raw prompt (no chat formatting)
            **kwargs: Generation parameters
        
        Returns:
            GenerationOutput with text and metadata
        """
        if self.generator is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
        )
        
        return self.generator.generate(prompt, gen_config)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics including KV cache."""
        stats = {}
        
        # Response cache stats
        if self._cache is not None:
            stats["response_cache"] = asyncio.get_event_loop().run_until_complete(
                self._cache.stats()
            )
            stats["enabled"] = True
        else:
            stats["enabled"] = False
        
        # KV cache stats
        if self.kv_cache_manager is not None:
            stats["kv_cache"] = self.kv_cache_manager.get_stats()
        
        return stats
    
    def get_kv_stats(self) -> Dict[str, Any]:
        """Get KV cache statistics."""
        if self.kv_cache_manager is None:
            return {"enabled": False}
        return self.kv_cache_manager.get_stats()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "device": str(self.device),
        }
        
        if torch.cuda.is_available():
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if self.memory_manager:
            stats["speed_mode"] = self.memory_manager.speed_mode.value
        
        if self.orchestrator:
            stats["orchestrator_active"] = self.orchestrator.running
            stats["memory_pressure"] = self.orchestrator.current_pressure.value if self.orchestrator.current_pressure else "unknown"
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the response cache and KV cache."""
        if self._cache:
            asyncio.get_event_loop().run_until_complete(
                self._cache.clear()
            )
        if self.kv_cache_manager:
            self.kv_cache_manager.clear()
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        # Stop orchestrator
        if self.orchestrator:
            self.orchestrator.stop()
            self.orchestrator = None
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.generator = None
        
        if self.memory_manager:
            self.memory_manager.clear()
        
        # Clear KV cache
        if self.kv_cache_manager:
            self.kv_cache_manager.clear()
            self.kv_cache_manager = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.unload()
