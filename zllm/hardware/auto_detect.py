"""
Hardware auto-detection and system information.

Automatically detects available hardware (CUDA, MPS, CPU, ROCm) and provides
optimal configuration recommendations.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple
from enum import Enum


class DeviceType(str, Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"
    ROCM = "rocm"
    CPU = "cpu"


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    total_memory: int  # bytes
    free_memory: int  # bytes
    compute_capability: Optional[Tuple[int, int]] = None  # For CUDA
    device_type: DeviceType = DeviceType.CUDA
    
    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024 ** 3)
    
    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024 ** 3)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.free_memory_gb:.1f}GB free / {self.total_memory_gb:.1f}GB total)"


@dataclass
class SystemInfo:
    """System memory and CPU information."""
    total_ram: int  # bytes
    available_ram: int  # bytes
    cpu_count: int
    cpu_name: str
    os_name: str
    os_version: str
    python_version: str
    
    @property
    def total_ram_gb(self) -> float:
        return self.total_ram / (1024 ** 3)
    
    @property
    def available_ram_gb(self) -> float:
        return self.available_ram / (1024 ** 3)


@dataclass
class DeviceInfo:
    """Complete device information for the system."""
    best_device: DeviceType
    gpus: List[GPUInfo]
    system: SystemInfo
    cuda_available: bool
    mps_available: bool
    rocm_available: bool
    
    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0
    
    @property
    def total_gpu_memory(self) -> int:
        return sum(gpu.total_memory for gpu in self.gpus)
    
    @property
    def total_gpu_memory_gb(self) -> float:
        return self.total_gpu_memory / (1024 ** 3)
    
    def get_recommended_max_layers(self, layer_size_mb: float = 500) -> int:
        """
        Estimate how many model layers can fit in available memory.
        
        Args:
            layer_size_mb: Estimated size of one model layer in MB
        """
        if self.has_gpu:
            available_mb = sum(gpu.free_memory for gpu in self.gpus) / (1024 ** 2)
        else:
            available_mb = self.system.available_ram / (1024 ** 2)
        
        # Reserve 20% for overhead
        usable_mb = available_mb * 0.8
        return max(1, int(usable_mb / layer_size_mb))
    
    def get_recommended_quantization(self, model_size_gb: float) -> Optional[str]:
        """
        Recommend quantization based on available memory and model size.
        
        Args:
            model_size_gb: Full model size in GB (fp16)
        """
        if self.has_gpu:
            available_gb = sum(gpu.free_memory for gpu in self.gpus) / (1024 ** 3)
        else:
            available_gb = self.system.available_ram / (1024 ** 3)
        
        # Reserve memory for KV cache and overhead (roughly 30%)
        usable_gb = available_gb * 0.7
        
        if model_size_gb <= usable_gb:
            return None  # No quantization needed
        elif model_size_gb * 0.5 <= usable_gb:
            return "int8"  # 8-bit reduces to ~50%
        else:
            return "int4"  # 4-bit reduces to ~25%


class HardwareDetector:
    """Auto-detect hardware capabilities."""
    
    def __init__(self):
        self._device_info: Optional[DeviceInfo] = None
    
    def detect(self, force_refresh: bool = False) -> DeviceInfo:
        """
        Detect all available hardware.
        
        Args:
            force_refresh: Force re-detection even if cached
        
        Returns:
            DeviceInfo with complete hardware information
        """
        if self._device_info is not None and not force_refresh:
            return self._device_info
        
        system_info = self._get_system_info()
        cuda_available, cuda_gpus = self._detect_cuda()
        mps_available, mps_info = self._detect_mps()
        rocm_available, rocm_gpus = self._detect_rocm()
        
        # Combine all GPUs
        gpus = cuda_gpus + rocm_gpus
        if mps_info:
            gpus.append(mps_info)
        
        # Determine best device
        if cuda_available and cuda_gpus:
            best_device = DeviceType.CUDA
        elif mps_available:
            best_device = DeviceType.MPS
        elif rocm_available and rocm_gpus:
            best_device = DeviceType.ROCM
        else:
            best_device = DeviceType.CPU
        
        self._device_info = DeviceInfo(
            best_device=best_device,
            gpus=gpus,
            system=system_info,
            cuda_available=cuda_available,
            mps_available=mps_available,
            rocm_available=rocm_available,
        )
        
        return self._device_info
    
    def _get_system_info(self) -> SystemInfo:
        """Get system RAM and CPU information."""
        import psutil
        
        mem = psutil.virtual_memory()
        
        # Get CPU name
        cpu_name = platform.processor()
        if not cpu_name:
            try:
                if platform.system() == "Darwin":
                    cpu_name = subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        text=True
                    ).strip()
                elif platform.system() == "Linux":
                    with open("/proc/cpuinfo") as f:
                        for line in f:
                            if "model name" in line:
                                cpu_name = line.split(":")[1].strip()
                                break
            except Exception:
                cpu_name = "Unknown CPU"
        
        return SystemInfo(
            total_ram=mem.total,
            available_ram=mem.available,
            cpu_count=os.cpu_count() or 1,
            cpu_name=cpu_name,
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),
        )
    
    def _detect_cuda(self) -> Tuple[bool, List[GPUInfo]]:
        """Detect NVIDIA CUDA GPUs."""
        gpus = []
        try:
            import torch
            if not torch.cuda.is_available():
                return False, []
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                
                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory=total_mem,
                    free_memory=free_mem,
                    compute_capability=(props.major, props.minor),
                    device_type=DeviceType.CUDA,
                ))
            
            return True, gpus
        except ImportError:
            return False, []
        except Exception:
            return False, []
    
    def _detect_mps(self) -> Tuple[bool, Optional[GPUInfo]]:
        """Detect Apple Metal Performance Shaders (Apple Silicon)."""
        if platform.system() != "Darwin":
            return False, None
        
        try:
            import torch
            if not torch.backends.mps.is_available():
                return False, None
            
            # MPS doesn't expose memory directly, estimate from system
            import psutil
            mem = psutil.virtual_memory()
            
            # Apple Silicon shares memory - estimate GPU portion
            # Typically 60-75% can be used for GPU
            gpu_memory = int(mem.total * 0.7)
            
            return True, GPUInfo(
                index=0,
                name="Apple Silicon GPU",
                total_memory=gpu_memory,
                free_memory=int(gpu_memory * 0.8),  # Estimate
                device_type=DeviceType.MPS,
            )
        except ImportError:
            return False, None
        except Exception:
            return False, None
    
    def _detect_rocm(self) -> Tuple[bool, List[GPUInfo]]:
        """Detect AMD ROCm GPUs."""
        gpus = []
        try:
            import torch
            if not (hasattr(torch, "hip") and torch.cuda.is_available()):
                # ROCm uses CUDA API but with HIP backend
                return False, []
            
            # Check if it's actually ROCm
            if "rocm" not in torch.__version__.lower() and "hip" not in str(torch.version.hip).lower():
                return False, []
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                
                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory=total_mem,
                    free_memory=free_mem,
                    device_type=DeviceType.ROCM,
                ))
            
            return True, gpus
        except ImportError:
            return False, []
        except Exception:
            return False, []
    
    def print_summary(self) -> None:
        """Print a human-readable summary of detected hardware."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        info = self.detect()
        
        # System info
        console.print(Panel(
            f"[bold]OS:[/bold] {info.system.os_name} {info.system.os_version}\n"
            f"[bold]CPU:[/bold] {info.system.cpu_name} ({info.system.cpu_count} cores)\n"
            f"[bold]RAM:[/bold] {info.system.available_ram_gb:.1f}GB available / {info.system.total_ram_gb:.1f}GB total\n"
            f"[bold]Python:[/bold] {info.system.python_version}",
            title="🖥️  System Information",
            border_style="blue"
        ))
        
        # GPU info
        if info.has_gpu:
            table = Table(title="🎮 GPU Devices")
            table.add_column("Device", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Memory", style="yellow")
            table.add_column("Type", style="magenta")
            
            for gpu in info.gpus:
                table.add_row(
                    f"#{gpu.index}",
                    gpu.name,
                    f"{gpu.free_memory_gb:.1f}GB / {gpu.total_memory_gb:.1f}GB",
                    gpu.device_type.value.upper(),
                )
            
            console.print(table)
        else:
            console.print("[yellow]No GPU detected. Using CPU for inference.[/yellow]")
        
        # Recommendation
        console.print(f"\n[bold green]✓ Best device:[/bold green] {info.best_device.value.upper()}")


# Global detector instance
_detector = HardwareDetector()


def detect_hardware(force_refresh: bool = False) -> DeviceInfo:
    """Convenience function to detect hardware."""
    return _detector.detect(force_refresh)


def get_best_device() -> str:
    """Get the best available device string for PyTorch."""
    info = _detector.detect()
    return info.best_device.value
