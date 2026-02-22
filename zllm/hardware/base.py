"""
Base hardware backend interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import torch


class HardwareBackend(ABC):
    """Abstract base class for hardware backends."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass
    
    @abstractmethod
    def get_device(self, index: int = 0) -> torch.device:
        """Get a torch device for this backend."""
        pass
    
    @abstractmethod
    def get_memory_info(self, device_index: int = 0) -> tuple[int, int]:
        """Get (free, total) memory in bytes."""
        pass
    
    @abstractmethod
    def empty_cache(self) -> None:
        """Clear GPU cache to free memory."""
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device operations."""
        pass
    
    def to_device(self, tensor: torch.Tensor, device_index: int = 0) -> torch.Tensor:
        """Move tensor to this device."""
        return tensor.to(self.get_device(device_index))


class CUDABackend(HardwareBackend):
    """NVIDIA CUDA backend."""
    
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def get_device(self, index: int = 0) -> torch.device:
        return torch.device(f"cuda:{index}")
    
    def get_memory_info(self, device_index: int = 0) -> tuple[int, int]:
        return torch.cuda.mem_get_info(device_index)
    
    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
    
    def synchronize(self) -> None:
        torch.cuda.synchronize()


class MPSBackend(HardwareBackend):
    """Apple Metal Performance Shaders backend."""
    
    def is_available(self) -> bool:
        return torch.backends.mps.is_available()
    
    def get_device(self, index: int = 0) -> torch.device:
        return torch.device("mps")
    
    def get_memory_info(self, device_index: int = 0) -> tuple[int, int]:
        # MPS doesn't expose memory info directly
        import psutil
        mem = psutil.virtual_memory()
        # Estimate GPU memory as 70% of system RAM for Apple Silicon
        total = int(mem.total * 0.7)
        free = int(mem.available * 0.7)
        return free, total
    
    def empty_cache(self) -> None:
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    def synchronize(self) -> None:
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()


class CPUBackend(HardwareBackend):
    """CPU fallback backend."""
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def get_device(self, index: int = 0) -> torch.device:
        return torch.device("cpu")
    
    def get_memory_info(self, device_index: int = 0) -> tuple[int, int]:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available, mem.total
    
    def empty_cache(self) -> None:
        import gc
        gc.collect()
    
    def synchronize(self) -> None:
        pass  # No-op for CPU


def get_backend(device_type: str = "auto") -> HardwareBackend:
    """
    Get the appropriate hardware backend.
    
    Args:
        device_type: One of "auto", "cuda", "mps", "cpu"
    
    Returns:
        HardwareBackend instance
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            return CUDABackend()
        elif torch.backends.mps.is_available():
            return MPSBackend()
        else:
            return CPUBackend()
    elif device_type == "cuda":
        return CUDABackend()
    elif device_type == "mps":
        return MPSBackend()
    else:
        return CPUBackend()
