"""
Model loader with SafeTensors and HuggingFace Hub support.

Supports layer-wise loading for memory efficiency.
Supports multiple quantization backends: bitsandbytes, AWQ, GPTQ, GGUF.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator, Tuple, Union
from dataclasses import dataclass
import torch
from torch import nn

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# AWQ support
try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False

# GPTQ support
try:
    from auto_gptq import AutoGPTQForCausalLM
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False

# GGUF/llama.cpp support
try:
    from llama_cpp import Llama
    from zllm.backends.gguf import GGUFModel, load_gguf, GGUF_AVAILABLE
except ImportError:
    GGUF_AVAILABLE = False
    GGUFModel = None
    load_gguf = None

from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model_path: Path
    config: Dict[str, Any]
    num_layers: int
    hidden_size: int
    vocab_size: int
    model_type: str
    total_params: int
    size_bytes: int
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)
    
    @property
    def params_billions(self) -> float:
        return self.total_params / 1e9


class ModelLoader:
    """
    Load models from HuggingFace Hub or local paths.
    
    Supports:
    - SafeTensors format (recommended)
    - PyTorch .bin format
    - Layer-wise loading for large models
    - Automatic model type detection
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        trust_remote_code: bool = False,
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code in model repos
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "zllm" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trust_remote_code = trust_remote_code
        self.api = HfApi()
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Get information about a model without loading it.
        
        Args:
            model_id: HuggingFace model ID or local path
        """
        # Resolve model path
        if Path(model_id).exists():
            model_path = Path(model_id)
        else:
            model_path = self._download_config(model_id)
        
        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Extract model info
        model_type = config.get("model_type", "unknown")
        hidden_size = config.get("hidden_size", config.get("n_embd", 0))
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        vocab_size = config.get("vocab_size", 0)
        
        # Estimate parameters and size
        total_params = self._estimate_params(config)
        size_bytes = total_params * 2  # Assuming fp16
        
        return ModelInfo(
            model_id=model_id,
            model_path=model_path,
            config=config,
            num_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            model_type=model_type,
            total_params=total_params,
            size_bytes=size_bytes,
        )
    
    def _estimate_params(self, config: Dict[str, Any]) -> int:
        """Estimate total parameters from config."""
        hidden_size = config.get("hidden_size", config.get("n_embd", 0))
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        vocab_size = config.get("vocab_size", 0)
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        num_heads = config.get("num_attention_heads", config.get("n_head", 0))
        
        if hidden_size == 0:
            return 0
        
        # Embedding params
        embed_params = vocab_size * hidden_size * 2  # input + output embeddings
        
        # Attention params per layer
        attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        
        # FFN params per layer
        ffn_params = 2 * hidden_size * intermediate_size + hidden_size * intermediate_size
        
        # Layer norm params
        ln_params = 4 * hidden_size  # 2 layer norms per layer
        
        # Total
        layer_params = attn_params + ffn_params + ln_params
        total = embed_params + (layer_params * num_layers)
        
        return total
    
    def _download_config(self, model_id: str) -> Path:
        """Download just the config file."""
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=self.cache_dir,
            )
            return Path(config_path).parent
        except Exception as e:
            raise RuntimeError(f"Failed to download config for {model_id}: {e}")
    
    def download_model(
        self,
        model_id: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
    ) -> Path:
        """
        Download a complete model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID
            revision: Git revision (branch, tag, commit)
            allow_patterns: File patterns to download
        
        Returns:
            Path to downloaded model directory
        """
        if allow_patterns is None:
            # Only download essential files
            allow_patterns = [
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "tokenizer*",
            ]
        
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=self.cache_dir,
            allow_patterns=allow_patterns,
        )
        
        return Path(model_path)
    
    def load_tokenizer(self, model_id: str) -> Any:
        """Load the tokenizer for a model."""
        return AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
    
    def detect_backend(self, model_id: str) -> str:
        """
        Auto-detect the best backend for a model based on its name/path.
        
        Returns:
            One of: 'awq', 'gptq', 'gguf', 'bitsandbytes'
        """
        model_id_lower = model_id.lower()
        
        # Check for GGUF file
        if model_id_lower.endswith('.gguf') or Path(model_id).suffix == '.gguf':
            if GGUF_AVAILABLE:
                return 'gguf'
            else:
                raise ImportError("GGUF file detected but llama-cpp-python is not installed. "
                                "Install with: pip install llama-cpp-python")
        
        # Check for AWQ model
        if 'awq' in model_id_lower or '-awq' in model_id_lower:
            if AWQ_AVAILABLE:
                return 'awq'
            else:
                raise ImportError("AWQ model detected but auto-awq is not installed. "
                                "Install with: pip install autoawq")
        
        # Check for GPTQ model
        if 'gptq' in model_id_lower or '-gptq' in model_id_lower:
            if GPTQ_AVAILABLE:
                return 'gptq'
            else:
                raise ImportError("GPTQ model detected but auto-gptq is not installed. "
                                "Install with: pip install auto-gptq")
        
        # Default to bitsandbytes
        return 'bitsandbytes'
    
    def get_available_backends(self) -> Dict[str, bool]:
        """Return availability status of all backends."""
        return {
            'bitsandbytes': True,  # Always available via transformers
            'awq': AWQ_AVAILABLE,
            'gptq': GPTQ_AVAILABLE,
            'gguf': GGUF_AVAILABLE,
        }
    
    def load_model_full(
        self,
        model_id: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float16,
        quantization: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Union[nn.Module, Any]:
        """
        Load a complete model with support for multiple backends.
        
        Args:
            model_id: HuggingFace model ID, local path, or GGUF file path
            device_map: Device mapping strategy
            dtype: Data type for model weights
            quantization: Quantization mode for bitsandbytes ('int8', 'int4')
            backend: Force a specific backend ('awq', 'gptq', 'gguf', 'bitsandbytes')
                     If None, auto-detects from model name
        
        Returns:
            Loaded model (type depends on backend)
        """
        # Auto-detect backend if not specified
        if backend is None:
            backend = self.detect_backend(model_id)
        
        # Load based on backend
        if backend == 'gguf':
            return self.load_model_gguf(model_id)
        
        if backend == 'awq':
            return self.load_model_awq(model_id, device_map)
        
        if backend == 'gptq':
            return self.load_model_gptq(model_id, device_map)
        
        # Default: bitsandbytes via transformers
        load_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype,
            "device_map": device_map,
        }
        
        if quantization == "int8":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                load_kwargs["load_in_8bit"] = True
        elif quantization == "int4":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                load_kwargs["load_in_4bit"] = True
        
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)
    
    def load_model_awq(
        self,
        model_id: str,
        device_map: str = "auto",
    ) -> Any:
        """
        Load an AWQ quantized model.
        
        AWQ (Activation-aware Weight Quantization) provides fast inference
        with 4-bit quantization. Typically 2x faster than bitsandbytes.
        
        Args:
            model_id: HuggingFace model ID for AWQ model
                     (e.g., "TheBloke/Llama-2-7B-Chat-AWQ")
            device_map: Device mapping strategy
        
        Returns:
            AWQ model ready for inference
        """
        if not AWQ_AVAILABLE:
            raise ImportError(
                "auto-awq is required for AWQ models. "
                "Install with: pip install autoawq"
            )
        
        model = AutoAWQForCausalLM.from_quantized(
            model_id,
            fuse_layers=True,  # Fused layers for faster inference
            trust_remote_code=self.trust_remote_code,
            safetensors=True,
        )
        return model
    
    def load_model_gptq(
        self,
        model_id: str,
        device_map: str = "auto",
    ) -> Any:
        """
        Load a GPTQ quantized model.
        
        GPTQ provides accurate 4-bit quantization with good inference speed.
        
        Args:
            model_id: HuggingFace model ID for GPTQ model
                     (e.g., "TheBloke/Llama-2-7B-Chat-GPTQ")
            device_map: Device mapping strategy
        
        Returns:
            GPTQ model ready for inference
        """
        if not GPTQ_AVAILABLE:
            raise ImportError(
                "auto-gptq is required for GPTQ models. "
                "Install with: pip install auto-gptq"
            )
        
        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            device="cuda:0",
            trust_remote_code=self.trust_remote_code,
            use_safetensors=True,
        )
        return model
    
    def load_model_gguf(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> Any:
        """
        Load a GGUF model using llama.cpp.
        
        GGUF models are highly optimized and can run on CPU or GPU.
        Typically 3-4x faster than bitsandbytes.
        
        Args:
            model_path: Path to .gguf file OR HuggingFace repo
                        (e.g., "Qwen/Qwen2-7B-Instruct-GGUF")
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            verbose: Enable verbose output
        
        Returns:
            GGUFModel ready for inference
        """
        if not GGUF_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models.\n"
                "Install with: pip install llama-cpp-python\n"
                "For GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )
        
        return load_gguf(
            model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            cache_dir=self.cache_dir,
        )
    
    def iter_layers_safetensors(
        self,
        model_path: Path,
        device: str = "cpu",
    ) -> Generator[Tuple[str, Dict[str, torch.Tensor]], None, None]:
        """
        Iterate over model layers from SafeTensors files.
        
        Yields one layer at a time for memory efficiency.
        
        Args:
            model_path: Path to model directory
            device: Device to load tensors to
        
        Yields:
            (layer_name, state_dict) tuples
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required for layer-wise loading")
        
        # Find all safetensors files
        st_files = list(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        # Build index of which tensors are in which files
        tensor_index: Dict[str, Path] = {}
        for st_file in st_files:
            with safe_open(st_file, framework="pt", device=device) as f:
                for key in f.keys():
                    tensor_index[key] = st_file
        
        # Group tensors by layer
        layer_tensors: Dict[str, List[str]] = {}
        for tensor_name in tensor_index:
            # Parse layer number from tensor name
            # Common patterns: "model.layers.0.self_attn.q_proj.weight"
            parts = tensor_name.split(".")
            layer_key = None
            
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        layer_key = f"layers.{layer_num}"
                        break
                    except ValueError:
                        continue
            
            if layer_key is None:
                layer_key = "other"  # Embeddings, final norm, etc.
            
            if layer_key not in layer_tensors:
                layer_tensors[layer_key] = []
            layer_tensors[layer_key].append(tensor_name)
        
        # Yield layers in order
        layer_keys = sorted(
            [k for k in layer_tensors if k.startswith("layers.")],
            key=lambda x: int(x.split(".")[1])
        )
        
        # First yield non-layer tensors (embeddings)
        if "other" in layer_tensors:
            state_dict = {}
            for tensor_name in layer_tensors["other"]:
                st_file = tensor_index[tensor_name]
                with safe_open(st_file, framework="pt", device=device) as f:
                    state_dict[tensor_name] = f.get_tensor(tensor_name)
            yield "embeddings", state_dict
        
        # Yield each layer
        for layer_key in layer_keys:
            state_dict = {}
            for tensor_name in layer_tensors[layer_key]:
                st_file = tensor_index[tensor_name]
                with safe_open(st_file, framework="pt", device=device) as f:
                    state_dict[tensor_name] = f.get_tensor(tensor_name)
            yield layer_key, state_dict
    
    def get_safetensors_metadata(self, model_path: Path) -> Dict[str, Any]:
        """Get metadata from SafeTensors files."""
        if not SAFETENSORS_AVAILABLE:
            return {}
        
        st_files = list(model_path.glob("*.safetensors"))
        if not st_files:
            return {}
        
        metadata = {}
        for st_file in st_files:
            with safe_open(st_file, framework="pt") as f:
                if hasattr(f, "metadata"):
                    file_metadata = f.metadata()
                    if file_metadata:
                        metadata[st_file.name] = file_metadata
        
        return metadata


class ModelRegistry:
    """
    Registry for managing downloaded and available models.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.models_dir = data_dir / "models"
        self.registry_file = data_dir / "model_registry.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self._registry = json.load(f)
    
    def _save_registry(self) -> None:
        """Save the registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self._registry, f, indent=2)
    
    def add_model(
        self,
        model_id: str,
        path: Path,
        info: ModelInfo,
    ) -> None:
        """Add a model to the registry."""
        self._registry[model_id] = {
            "path": str(path),
            "model_type": info.model_type,
            "num_layers": info.num_layers,
            "hidden_size": info.hidden_size,
            "size_gb": info.size_gb,
            "params_billions": info.params_billions,
        }
        self._save_registry()
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id in self._registry:
            del self._registry[model_id]
            self._save_registry()
            return True
        return False
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model from the registry."""
        return self._registry.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return [
            {"model_id": k, **v}
            for k, v in self._registry.items()
        ]
    
    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded."""
        return model_id in self._registry
