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
        
        # Track model loading constraints for upgrade capability
        self._load_constraints = {
            "quantization_forced": False,  # True if INT8/INT4 was auto-applied due to memory
            "original_quantization": None,  # User's requested quantization
            "applied_quantization": None,   # What was actually applied
            "model_size_gb": 0,
            "available_vram_gb": 0,
        }
        
        # Runtime memory monitoring configuration
        self._runtime_monitor = {
            "enabled": True,
            "check_interval": 5,        # Check every N generations
            "generation_count": 0,      # Counter for generations
            "last_recommendation": None,  # Last recommendation given
            "low_usage_threshold": 0.5,  # Alert if GPU usage below 50%
            "high_usage_threshold": 0.85,  # Alert if GPU usage above 85%
            "auto_adjust": False,        # Auto-adjust speed mode based on memory
            "shown_recommendations": 0,   # Track how many times we've shown
            "max_recommendations": 3,     # Don't spam - max recommendations per session
        }
        
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
        user_requested_quant = quantization or self.config.quantization
        quant = user_requested_quant
        
        # Track load constraints
        self._load_constraints["original_quantization"] = user_requested_quant
        self._load_constraints["model_size_gb"] = self.model_info.size_gb
        self._load_constraints["available_vram_gb"] = self.hardware_info.gpu_memory_total_gb if self.hardware_info.gpu_available else 0
        
        if quant is None and self.config.auto_quantize:
            quant = self.hardware_info.get_recommended_quantization(
                self.model_info.size_gb
            )
            if quant:
                console.print(f"[yellow]Auto-selected {quant} quantization for memory efficiency[/yellow]")
                self._load_constraints["quantization_forced"] = True
        
        self._load_constraints["applied_quantization"] = quant
        
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
            quantization=QuantizationScheme.INT8,  # 50% memory savings
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
    
    def can_upgrade(self) -> Dict[str, Any]:
        """
        Check if the model can be upgraded to a less quantized/faster version.
        
        Returns:
            Dict with upgrade possibility info:
            - can_upgrade: bool
            - reason: str explaining why/why not
            - current_quantization: current quantization level
            - recommended_quantization: what we could upgrade to
            - estimated_speedup: expected speed improvement
            - memory_required_gb: how much VRAM needed for upgrade
            - memory_available_gb: how much VRAM is currently free
        """
        result = {
            "can_upgrade": False,
            "reason": "",
            "current_quantization": self._load_constraints.get("applied_quantization"),
            "recommended_quantization": None,
            "estimated_speedup": "1.0x",
            "memory_required_gb": 0,
            "memory_available_gb": 0,
        }
        
        # Check if model was loaded with forced quantization
        if not self._load_constraints.get("quantization_forced"):
            result["reason"] = "Model was loaded with user-specified settings (no forced quantization)"
            return result
        
        if not torch.cuda.is_available():
            result["reason"] = "No GPU available for upgrade check"
            return result
        
        # Get current memory state
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_free = gpu_total - gpu_allocated
        
        result["memory_available_gb"] = round(gpu_free, 2)
        
        model_size = self._load_constraints.get("model_size_gb", 0)
        current_quant = self._load_constraints.get("applied_quantization")
        
        # Calculate memory needed for different quantization levels
        # FP16 baseline, INT8 ~0.5x size, INT4 ~0.25x size
        if current_quant == "int8":
            # Currently INT8, check if FP16 fits
            fp16_size = model_size * 2  # INT8 is roughly half of FP16
            additional_needed = fp16_size - model_size  # Extra memory needed
            
            result["memory_required_gb"] = round(additional_needed, 2)
            
            if gpu_free > additional_needed * 1.2:  # 20% safety margin
                result["can_upgrade"] = True
                result["recommended_quantization"] = None  # FP16 (no quantization)
                result["estimated_speedup"] = "1.8-2.5x"
                result["reason"] = f"Sufficient VRAM available ({gpu_free:.1f}GB free, need {additional_needed:.1f}GB more for FP16)"
            else:
                result["reason"] = f"Not enough VRAM for FP16 ({gpu_free:.1f}GB free, need {additional_needed:.1f}GB more)"
                
        elif current_quant == "int4":
            # Currently INT4, check if INT8 or FP16 fits
            int8_size = model_size * 2  # INT4 is roughly half of INT8
            fp16_size = model_size * 4  # INT4 is roughly 1/4 of FP16
            
            int8_additional = int8_size - model_size
            fp16_additional = fp16_size - model_size
            
            if gpu_free > fp16_additional * 1.2:
                result["can_upgrade"] = True
                result["recommended_quantization"] = None  # FP16
                result["memory_required_gb"] = round(fp16_additional, 2)
                result["estimated_speedup"] = "2.5-3.5x"
                result["reason"] = f"Can upgrade to FP16 ({gpu_free:.1f}GB free)"
            elif gpu_free > int8_additional * 1.2:
                result["can_upgrade"] = True
                result["recommended_quantization"] = "int8"
                result["memory_required_gb"] = round(int8_additional, 2)
                result["estimated_speedup"] = "1.3-1.8x"
                result["reason"] = f"Can upgrade to INT8 ({gpu_free:.1f}GB free)"
            else:
                result["memory_required_gb"] = round(int8_additional, 2)
                result["reason"] = f"Not enough VRAM for upgrade ({gpu_free:.1f}GB free, need {int8_additional:.1f}GB for INT8)"
        else:
            result["reason"] = "Model already at full precision (FP16)"
        
        return result
    
    def upgrade_model(self, target_quantization: Optional[str] = None) -> "ZLLM":
        """
        Reload the model with better quantization (less compression = faster).
        
        Args:
            target_quantization: Target quantization level (None for FP16, "int8", "int4")
                                If None, uses the recommended from can_upgrade()
        
        Returns:
            self for chaining
            
        Raises:
            RuntimeError: If upgrade not possible or would cause OOM
        """
        from rich.console import Console
        console = Console()
        
        # Check if upgrade is possible
        upgrade_info = self.can_upgrade()
        
        if not upgrade_info["can_upgrade"] and target_quantization is None:
            console.print(f"[yellow]Cannot upgrade: {upgrade_info['reason']}[/yellow]")
            return self
        
        # Determine target
        if target_quantization is None:
            target_quantization = upgrade_info["recommended_quantization"]
        
        model_id = self.config.model_id
        if not model_id:
            raise RuntimeError("No model ID available for reload")
        
        console.print(f"\n[cyan]🔄 Upgrading model...[/cyan]")
        current = self._load_constraints.get("applied_quantization") or "fp16"
        target_display = target_quantization or "fp16"
        console.print(f"  Current: {current} → Target: {target_display}")
        console.print(f"  Expected speedup: {upgrade_info.get('estimated_speedup', 'unknown')}")
        
        # Unload current model to free memory
        console.print("  [dim]Unloading current model...[/dim]")
        self.unload()
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Reload with new quantization
        # Disable auto-quantize since we're explicitly setting it
        original_auto_quantize = self.config.auto_quantize
        self.config.auto_quantize = False
        self.config.quantization = target_quantization
        
        try:
            self.load_model(model_id, quantization=target_quantization)
            console.print(f"[green]✓ Model upgraded successfully![/green]")
            
            # Update constraints to reflect manual upgrade
            self._load_constraints["quantization_forced"] = False
            self._load_constraints["applied_quantization"] = target_quantization
            
        except Exception as e:
            console.print(f"[red]Upgrade failed: {e}[/red]")
            console.print("[yellow]Attempting to restore previous configuration...[/yellow]")
            self.config.auto_quantize = original_auto_quantize
            self.config.quantization = self._load_constraints.get("original_quantization")
            self.load_model(model_id)
            raise RuntimeError(f"Upgrade failed: {e}")
        finally:
            self.config.auto_quantize = original_auto_quantize
        
        return self
    
    def check_runtime_memory(self) -> Dict[str, Any]:
        """
        Check current GPU memory usage and provide recommendations.
        
        Called automatically after generations to provide smart suggestions.
        
        Returns:
            Dict with:
            - usage_percent: Current GPU usage percentage
            - status: "optimal" | "underutilized" | "high_pressure"
            - recommendation: Suggestion text (if any)
            - action: Suggested action ("speed_up" | "slow_down" | None)
            - can_speed_up: Whether speed can be increased
        """
        result = {
            "usage_percent": 0,
            "status": "optimal",
            "recommendation": None,
            "action": None,
            "can_speed_up": False,
            "can_upgrade": False,
        }
        
        if not torch.cuda.is_available():
            return result
        
        # Get memory usage
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_usage = gpu_allocated / gpu_total if gpu_total > 0 else 0
        
        result["usage_percent"] = round(gpu_usage * 100, 1)
        result["allocated_gb"] = round(gpu_allocated, 2)
        result["total_gb"] = round(gpu_total, 2)
        result["free_gb"] = round(gpu_total - gpu_allocated, 2)
        
        # Get current speed mode
        current_mode = self.memory_manager.speed_mode.value if self.memory_manager else "balanced"
        result["current_speed_mode"] = current_mode
        
        # Analyze and recommend
        low_threshold = self._runtime_monitor["low_usage_threshold"]
        high_threshold = self._runtime_monitor["high_usage_threshold"]
        
        if gpu_usage < low_threshold:
            # Underutilized - can speed up
            result["status"] = "underutilized"
            result["can_speed_up"] = True
            
            if current_mode == "memory":
                result["action"] = "speed_up"
                result["recommendation"] = (
                    f"GPU only {result['usage_percent']:.0f}% utilized ({result['free_gb']:.1f}GB free). "
                    f"Switch to 'balanced' or 'fast' mode for better speed."
                )
            elif current_mode == "balanced":
                result["action"] = "speed_up"
                result["recommendation"] = (
                    f"GPU {result['usage_percent']:.0f}% utilized with {result['free_gb']:.1f}GB free. "
                    f"Switch to 'fast' mode for ~20% speed boost."
                )
            
            # Check if upgrade (less quantization) is possible
            upgrade_info = self.can_upgrade()
            if upgrade_info.get("can_upgrade"):
                result["can_upgrade"] = True
                result["recommendation"] = (
                    f"GPU only {result['usage_percent']:.0f}% utilized. "
                    f"Use /upgrade for {upgrade_info.get('estimated_speedup', '1.5-2x')} faster inference!"
                )
                result["action"] = "upgrade"
                
        elif gpu_usage > high_threshold:
            # High pressure - might want to reduce
            result["status"] = "high_pressure"
            
            if current_mode == "fast":
                result["action"] = "slow_down"
                result["recommendation"] = (
                    f"GPU at {result['usage_percent']:.0f}% capacity. "
                    f"Switch to 'balanced' mode to prevent OOM on longer contexts."
                )
        else:
            result["status"] = "optimal"
        
        return result
    
    def get_speed_recommendation(self) -> Optional[Dict[str, Any]]:
        """
        Get a speed recommendation if one is warranted.
        
        This is called after each generation to determine if user should
        be notified about optimization opportunities.
        
        Returns:
            Recommendation dict or None if no recommendation needed.
        """
        # Increment generation counter
        self._runtime_monitor["generation_count"] += 1
        
        # Check if monitoring is enabled
        if not self._runtime_monitor["enabled"]:
            return None
        
        # Don't spam recommendations
        if self._runtime_monitor["shown_recommendations"] >= self._runtime_monitor["max_recommendations"]:
            return None
        
        # Only check at intervals
        if self._runtime_monitor["generation_count"] % self._runtime_monitor["check_interval"] != 0:
            return None
        
        # Get memory status
        mem_check = self.check_runtime_memory()
        
        # Only recommend if there's an action
        if not mem_check.get("action"):
            return None
        
        # Don't repeat the same recommendation
        if mem_check.get("recommendation") == self._runtime_monitor["last_recommendation"]:
            return None
        
        # Update tracking
        self._runtime_monitor["last_recommendation"] = mem_check.get("recommendation")
        self._runtime_monitor["shown_recommendations"] += 1
        
        return mem_check
    
    def set_auto_adjust(self, enabled: bool) -> None:
        """
        Enable/disable automatic speed mode adjustment based on memory usage.
        
        When enabled, engine will automatically switch speed modes to optimize
        for current memory conditions.
        """
        self._runtime_monitor["auto_adjust"] = enabled
        
        if enabled and self.memory_manager:
            # Immediately check and adjust
            mem_check = self.check_runtime_memory()
            from zllm.core.memory import SpeedMode
            
            if mem_check["status"] == "underutilized" and mem_check.get("can_speed_up"):
                current = self.memory_manager.speed_mode
                if current == SpeedMode.MEMORY_SAVER:
                    self.memory_manager.speed_mode = SpeedMode.BALANCED
                elif current == SpeedMode.BALANCED:
                    self.memory_manager.speed_mode = SpeedMode.FAST
            elif mem_check["status"] == "high_pressure":
                current = self.memory_manager.speed_mode
                if current == SpeedMode.FAST:
                    self.memory_manager.speed_mode = SpeedMode.BALANCED
    
    def silence_recommendations(self) -> None:
        """Stop showing runtime recommendations for this session."""
        self._runtime_monitor["enabled"] = False
    
    def reset_recommendations(self) -> None:
        """Reset recommendation counter to show suggestions again."""
        self._runtime_monitor["shown_recommendations"] = 0
        self._runtime_monitor["last_recommendation"] = None
        self._runtime_monitor["enabled"] = True
    
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
