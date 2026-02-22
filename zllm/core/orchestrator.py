"""
Intelligent Orchestrator for Dynamic VRAM Management.

This is zllm's secret sauce - automatically optimizes VRAM usage for maximum speed
while preventing out-of-memory errors.

Key Features:
1. Real-time VRAM monitoring
2. Dynamic layer budget adjustment
3. Auto speed mode selection
4. Predictive prefetching
5. Memory pressure handling
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Callable, Any
import torch

from zllm.hardware.base import get_backend
from zllm.hardware.auto_detect import detect_hardware
from zllm.core.memory import MemoryManager, SpeedMode, LayerCache


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = "low"           # < 50% used, can cache more
    NORMAL = "normal"     # 50-75% used, optimal
    HIGH = "high"         # 75-90% used, be careful
    CRITICAL = "critical" # > 90% used, evict immediately


@dataclass
class OrchestratorStats:
    """Statistics from the orchestrator."""
    current_layers_cached: int
    max_layers_possible: int
    vram_used_gb: float
    vram_total_gb: float
    vram_utilization: float
    memory_pressure: MemoryPressure
    speed_mode: SpeedMode
    estimated_tokens_per_sec: float
    adjustments_made: int


class IntelligentOrchestrator:
    """
    Intelligent VRAM orchestrator for maximum inference speed.
    
    Instead of static memory allocation, this dynamically adjusts
    based on real-time VRAM availability:
    
    - VRAM available? → Cache more layers → Faster inference
    - VRAM tight? → Evict layers → Prevent OOM
    - Other apps using GPU? → Adapt automatically
    
    Usage:
        orchestrator = IntelligentOrchestrator(memory_manager)
        orchestrator.start()  # Background monitoring
        
        # During inference, orchestrator auto-tunes
        with orchestrator.inference_context():
            for layer in layers:
                layer = memory_manager.get_layer(layer_name)
                output = layer(input)
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        target_utilization: float = 0.80,  # Target 80% VRAM usage
        min_free_vram_mb: int = 512,       # Always keep 512MB free
        check_interval: float = 0.5,        # Check every 500ms
        enable_predictive: bool = True,     # Predictive prefetching
    ):
        """
        Initialize the orchestrator.
        
        Args:
            memory_manager: The memory manager to orchestrate
            target_utilization: Target VRAM utilization (0.0-1.0)
            min_free_vram_mb: Minimum free VRAM to maintain
            check_interval: How often to check memory (seconds)
            enable_predictive: Enable predictive layer prefetching
        """
        self.mm = memory_manager
        self.target_utilization = target_utilization
        self.min_free_vram_mb = min_free_vram_mb
        self.check_interval = check_interval
        self.enable_predictive = enable_predictive
        
        self.backend = get_backend("auto")
        self.hardware = detect_hardware()
        
        # State
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Stats
        self._adjustments_made = 0
        self._current_speed_mode = SpeedMode.BALANCED
        self._layer_access_history: List[str] = []
        
        # Callbacks
        self._on_pressure_change: Optional[Callable[[MemoryPressure], None]] = None
        
    def start(self) -> None:
        """Start the background orchestrator."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="zllm-orchestrator"
        )
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_and_adjust()
            except Exception as e:
                # Log but don't crash
                pass
            time.sleep(self.check_interval)
    
    def _get_memory_pressure(self) -> MemoryPressure:
        """Determine current memory pressure level."""
        if not self.backend.is_available():
            return MemoryPressure.NORMAL
        
        free, total = self.backend.get_memory_info()
        utilization = 1.0 - (free / total)
        free_mb = free / (1024 * 1024)
        
        if free_mb < self.min_free_vram_mb:
            return MemoryPressure.CRITICAL
        elif utilization > 0.90:
            return MemoryPressure.CRITICAL
        elif utilization > 0.75:
            return MemoryPressure.HIGH
        elif utilization > 0.50:
            return MemoryPressure.NORMAL
        else:
            return MemoryPressure.LOW
    
    def _check_and_adjust(self) -> None:
        """Check memory and adjust layer caching."""
        pressure = self._get_memory_pressure()
        
        with self._lock:
            if pressure == MemoryPressure.CRITICAL:
                # Emergency: evict layers immediately
                self._emergency_evict()
                self._current_speed_mode = SpeedMode.MEMORY_SAVER
                self._adjustments_made += 1
                
            elif pressure == MemoryPressure.HIGH:
                # High pressure: switch to memory saver mode
                if self._current_speed_mode != SpeedMode.MEMORY_SAVER:
                    self._reduce_cache()
                    self._current_speed_mode = SpeedMode.MEMORY_SAVER
                    self._adjustments_made += 1
                    
            elif pressure == MemoryPressure.LOW:
                # Low pressure: we can cache more for speed!
                if self._current_speed_mode != SpeedMode.FAST:
                    self._expand_cache()
                    self._current_speed_mode = SpeedMode.FAST
                    self._adjustments_made += 1
                    
            else:  # NORMAL
                if self._current_speed_mode != SpeedMode.BALANCED:
                    self._current_speed_mode = SpeedMode.BALANCED
        
        # Notify callback if registered
        if self._on_pressure_change:
            self._on_pressure_change(pressure)
    
    def _emergency_evict(self) -> None:
        """Emergency eviction when VRAM is critical."""
        if not self.mm.gpu_cache:
            return
        
        # Evict 30% of cached layers
        current_count = len(self.mm.gpu_cache)
        evict_count = max(1, int(current_count * 0.3))
        
        keys = self.mm.gpu_cache.keys()
        for key in keys[:evict_count]:
            self.mm.unload_layer(key)
        
        # Force garbage collection
        self.backend.empty_cache()
    
    def _reduce_cache(self) -> None:
        """Reduce cache size for high memory pressure."""
        if not self.mm.gpu_cache:
            return
        
        # Reduce to 60% capacity
        current_count = len(self.mm.gpu_cache)
        target_count = int(current_count * 0.6)
        evict_count = current_count - target_count
        
        if evict_count > 0:
            keys = self.mm.gpu_cache.keys()
            for key in keys[:evict_count]:
                self.mm.unload_layer(key)
    
    def _expand_cache(self) -> None:
        """Expand cache when VRAM is available."""
        if not self.mm.gpu_cache:
            return
        
        # Increase max layers by 25%
        current_max = self.mm.gpu_cache.max_layers
        new_max = int(current_max * 1.25)
        
        # Calculate available memory for expansion
        free, total = self.backend.get_memory_info()
        available_for_expansion = free - (self.min_free_vram_mb * 1024 * 1024)
        
        if available_for_expansion > 0 and self.mm.layer_info:
            # Estimate layer size from registered layers
            sample_layer = next(iter(self.mm.layer_info.values()))
            layer_size = sample_layer.size_bytes
            
            # How many more layers can we fit?
            extra_layers = available_for_expansion // layer_size
            new_max = min(new_max, len(self.mm.gpu_cache) + extra_layers)
        
        self.mm.gpu_cache.max_layers = new_max
    
    def auto_select_speed_mode(self, model_layers: int, layer_size_bytes: int) -> SpeedMode:
        """
        Automatically select the best speed mode based on available VRAM.
        
        Args:
            model_layers: Total layers in the model
            layer_size_bytes: Size of each layer
        
        Returns:
            Recommended SpeedMode
        """
        if not self.backend.is_available():
            return SpeedMode.MEMORY_SAVER
        
        free, total = self.backend.get_memory_info()
        
        # Calculate what we can fit
        usable_vram = free - (self.min_free_vram_mb * 1024 * 1024)
        
        # Full model size
        full_model_size = model_layers * layer_size_bytes
        
        # What fraction can we load?
        loadable_fraction = usable_vram / full_model_size
        
        if loadable_fraction >= 0.75:
            return SpeedMode.FAST        # Can load 75%+ → go fast!
        elif loadable_fraction >= 0.50:
            return SpeedMode.BALANCED    # Can load 50-75% → balanced
        else:
            return SpeedMode.MEMORY_SAVER # Less than 50% → memory mode
    
    def get_stats(self) -> OrchestratorStats:
        """Get current orchestrator statistics."""
        pressure = self._get_memory_pressure()
        
        vram_used_gb = 0.0
        vram_total_gb = 0.0
        vram_utilization = 0.0
        
        if self.backend.is_available():
            free, total = self.backend.get_memory_info()
            vram_used_gb = (total - free) / (1024 ** 3)
            vram_total_gb = total / (1024 ** 3)
            vram_utilization = 1.0 - (free / total)
        
        current_layers = len(self.mm.gpu_cache) if self.mm.gpu_cache else 0
        max_layers = self.mm.gpu_cache.max_layers if self.mm.gpu_cache else 0
        
        # Estimate tokens/sec based on cached layers
        # More cached layers = less IO = faster
        estimated_tps = 5 + (current_layers * 1.2)  # Rough estimate
        
        return OrchestratorStats(
            current_layers_cached=current_layers,
            max_layers_possible=max_layers,
            vram_used_gb=vram_used_gb,
            vram_total_gb=vram_total_gb,
            vram_utilization=vram_utilization,
            memory_pressure=pressure,
            speed_mode=self._current_speed_mode,
            estimated_tokens_per_sec=estimated_tps,
            adjustments_made=self._adjustments_made,
        )
    
    def record_layer_access(self, layer_name: str) -> None:
        """Record layer access for predictive prefetching."""
        if self.enable_predictive:
            self._layer_access_history.append(layer_name)
            # Keep last 1000 accesses
            if len(self._layer_access_history) > 1000:
                self._layer_access_history = self._layer_access_history[-500:]
    
    def predict_next_layers(self, current_layer: str, count: int = 3) -> List[str]:
        """
        Predict which layers will be needed next based on access patterns.
        
        This enables smarter prefetching - not just sequential, but based
        on actual usage patterns.
        """
        if not self.enable_predictive or not self._layer_access_history:
            return []
        
        # Find patterns after current layer
        predictions = {}
        for i, layer in enumerate(self._layer_access_history[:-1]):
            if layer == current_layer:
                next_layer = self._layer_access_history[i + 1]
                predictions[next_layer] = predictions.get(next_layer, 0) + 1
        
        # Sort by frequency
        sorted_predictions = sorted(predictions.items(), key=lambda x: -x[1])
        return [layer for layer, _ in sorted_predictions[:count]]
    
    def on_pressure_change(self, callback: Callable[[MemoryPressure], None]) -> None:
        """Register callback for memory pressure changes."""
        self._on_pressure_change = callback
    
    def inference_context(self):
        """Context manager for inference with orchestration."""
        return OrchestratorContext(self)
    
    def print_stats(self) -> None:
        """Print orchestrator statistics."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        stats = self.get_stats()
        
        # Pressure color
        pressure_colors = {
            MemoryPressure.LOW: "green",
            MemoryPressure.NORMAL: "blue",
            MemoryPressure.HIGH: "yellow",
            MemoryPressure.CRITICAL: "red",
        }
        pressure_color = pressure_colors[stats.memory_pressure]
        
        table = Table(title="🧠 Intelligent Orchestrator Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("VRAM Usage", f"{stats.vram_used_gb:.1f}GB / {stats.vram_total_gb:.1f}GB ({stats.vram_utilization*100:.0f}%)")
        table.add_row("Memory Pressure", f"[{pressure_color}]{stats.memory_pressure.value}[/{pressure_color}]")
        table.add_row("Speed Mode", stats.speed_mode.value)
        table.add_row("Layers Cached", f"{stats.current_layers_cached} / {stats.max_layers_possible}")
        table.add_row("Est. Speed", f"~{stats.estimated_tokens_per_sec:.0f} tok/s")
        table.add_row("Auto-adjustments", str(stats.adjustments_made))
        
        console.print(table)


class OrchestratorContext:
    """Context manager for orchestrated inference."""
    
    def __init__(self, orchestrator: IntelligentOrchestrator):
        self.orchestrator = orchestrator
        self._was_running = False
    
    def __enter__(self):
        self._was_running = self.orchestrator._running
        if not self._was_running:
            self.orchestrator.start()
        return self.orchestrator
    
    def __exit__(self, *args):
        # Keep running if it was already running
        if not self._was_running:
            self.orchestrator.stop()


# Convenience function
def create_orchestrated_memory_manager(
    device: str = "auto",
    auto_optimize: bool = True,
) -> tuple[MemoryManager, IntelligentOrchestrator]:
    """
    Create a memory manager with intelligent orchestration.
    
    Args:
        device: Target device
        auto_optimize: Start orchestrator immediately
    
    Returns:
        Tuple of (MemoryManager, IntelligentOrchestrator)
    """
    mm = MemoryManager(device=device)
    orchestrator = IntelligentOrchestrator(mm)
    
    if auto_optimize:
        orchestrator.start()
    
    return mm, orchestrator
