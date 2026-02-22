"""
Configuration management for zllm.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import os
import json


def get_default_cache_dir() -> Path:
    """Get the default cache directory for zllm."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.name == "posix":
        if "darwin" in os.uname().sysname.lower():  # macOS
            base = Path.home() / "Library" / "Caches"
        else:  # Linux
            base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    else:
        base = Path.home() / ".cache"
    
    return base / "zllm"


def get_default_data_dir() -> Path:
    """Get the default data directory for zllm."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.name == "posix":
        if "darwin" in os.uname().sysname.lower():  # macOS
            base = Path.home() / "Library" / "Application Support"
        else:  # Linux
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    else:
        base = Path.home() / ".local" / "share"
    
    return base / "zllm"


@dataclass
class ZLLMConfig:
    """Main configuration for zllm."""
    
    # Model settings
    model_id: Optional[str] = None
    model_path: Optional[Path] = None
    
    # Hardware settings
    device: Literal["auto", "cuda", "mps", "cpu", "rocm"] = "auto"
    device_map: Optional[str] = "auto"
    max_memory: Optional[dict] = None
    
    # Quantization settings
    quantization: Optional[Literal["int4", "int8", "none"]] = None
    auto_quantize: bool = True  # Auto-select quantization based on available memory
    
    # Memory management
    enable_layer_streaming: bool = True
    max_layers_in_memory: Optional[int] = None
    offload_to_cpu: bool = True
    
    # Speed vs Memory trade-off
    # "fast": Use more VRAM for speed (75% of available)
    # "balanced": Sweet spot - fast + efficient (60% of available) [DEFAULT]
    # "memory": Minimum memory, slower (40% of available)
    speed_mode: Literal["fast", "balanced", "memory"] = "balanced"
    
    # Generation defaults
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Cache settings
    enable_cache: bool = True
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.92
    cache_max_size: int = 1000
    
    # Flash Attention settings
    enable_flash_attention: bool = True  # Auto-enable best available backend
    flash_attention_backend: Optional[Literal["auto", "flash_attn", "sdpa", "chunked"]] = "auto"
    attention_sliding_window: Optional[int] = None  # For long context models
    
    # Speculative Decoding settings
    enable_speculative: bool = False  # Enable with --speculative flag
    draft_model_id: Optional[str] = None  # Smaller model for speculation
    num_speculative_tokens: int = 5  # Tokens to speculate per step
    speculative_acceptance: Literal["greedy", "sampling", "threshold"] = "sampling"
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    api_key_required: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])
    
    # Paths
    cache_dir: Path = field(default_factory=get_default_cache_dir)
    data_dir: Path = field(default_factory=get_default_data_dir)
    models_dir: Optional[Path] = None
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    def __post_init__(self):
        """Initialize paths and validate config."""
        if self.models_dir is None:
            self.models_dir = self.data_dir / "models"
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, path: Path) -> "ZLLMConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Convert path strings to Path objects
        for key in ["cache_dir", "data_dir", "models_dir", "model_path"]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save configuration to a JSON file."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_config_path(self) -> Path:
        """Get the path to the config file."""
        return self.data_dir / "config.json"
