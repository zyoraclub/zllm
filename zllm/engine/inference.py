"""
ZLLM Native Inference Engine.

Our own transformer inference implementation - no external dependencies.

Features:
- Layer-wise loading for memory efficiency
- Supports any GGUF model
- Custom CUDA/Triton kernels for acceleration
- PyTorch-native fallback operations
"""

import math
from typing import Optional, Dict, List, Tuple, Iterator, Any
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gguf_parser import GGUFParser, GGUFMetadata, GGUFTensor
from .quantization import dequantize_tensor

# Try to import CUDA kernels
try:
    from .cuda_kernels import (
        rms_norm_cuda,
        apply_rope_cuda,
        flash_attention_cuda,
        is_triton_available,
    )
    CUDA_KERNELS_AVAILABLE = is_triton_available()
except ImportError:
    CUDA_KERNELS_AVAILABLE = False
    rms_norm_cuda = None
    apply_rope_cuda = None
    flash_attention_cuda = None


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    max_seq_len: int = 4096
    max_batch_size: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    
    # Memory options
    layer_offload: bool = False  # Offload layers to CPU when not in use (legacy)
    kv_cache_quantize: bool = True  # Quantize KV cache
    
    # Layer Streaming - THE KEY TO RUNNING 7B ON 2-3GB VRAM
    layer_streaming: bool = True  # Enable layer streaming (load layers on demand)
    max_gpu_layers: Optional[int] = None  # Max layers on GPU (auto-calculated if None)
    speed_mode: str = "balanced"  # "fast", "balanced", "memory" 
    prefetch_layers: int = 2  # Number of layers to prefetch ahead
    
    # Performance options
    use_cuda_kernels: bool = True  # Use custom CUDA/Triton kernels when available
    use_flash_attention: bool = True  # Use Flash Attention


class RMSNorm(nn.Module):
    """RMS Normalization (used by LLaMA, Qwen, etc.)."""
    
    def __init__(self, dim: int, eps: float = 1e-6, use_cuda: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_cuda = use_cuda and CUDA_KERNELS_AVAILABLE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuda and x.is_cuda:
            return rms_norm_cuda(x, self.weight, self.eps)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_rope_cache(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE sin/cos cache."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    
    cos_cache = torch.cos(freqs)
    sin_cache = torch.sin(freqs)
    
    return cos_cache, sin_cache


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings (RoPE) with INTERLEAVED layout.
    
    LLaMA/Qwen use interleaved pairs: [x0, x1, x2, x3, ...] -> [r0, i0, r1, i1, ...]
    where (r, i) are rotated (real, imaginary) pairs.
    
    Args:
        q: Query tensor [B, H, T, D]
        k: Key tensor [B, H, T, D]
        cos: Cosine cache [T, D//2]
        sin: Sine cache [T, D//2]
        position_ids: Position indices [B, T]
    """
    # Get positions: cos/sin are [max_seq, dim//2], index to [B, T, dim//2]
    cos = cos[position_ids]  # (batch, seq_len, dim//2)
    sin = sin[position_ids]
    
    # CRITICAL: Match dtype to avoid FP32/FP16 mixing in attention
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    
    # Reshape for broadcasting: [B, 1, T, dim//2] broadcasts to [B, H, T, dim//2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Split into even and odd (interleaved pairs)
    q1, q2 = q[..., ::2], q[..., 1::2]  # real, imag parts
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Apply rotation with INTERLEAVED output (critical for correctness!)
    # Wrong: torch.cat([real, imag]) gives sequential layout
    # Right: interleave back to original positions
    q_rotated = torch.empty_like(q)
    k_rotated = torch.empty_like(k)
    
    q_rotated[..., ::2] = q1 * cos - q2 * sin   # rotated real -> even indices
    q_rotated[..., 1::2] = q1 * sin + q2 * cos  # rotated imag -> odd indices
    
    k_rotated[..., ::2] = k1 * cos - k2 * sin
    k_rotated[..., 1::2] = k1 * sin + k2 * cos
    
    return q_rotated, k_rotated


class TransformerLayer(nn.Module):
    """A single transformer layer."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        use_cuda_kernels: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.use_cuda_kernels = use_cuda_kernels and CUDA_KERNELS_AVAILABLE
        self.use_flash_attention = use_flash_attention and CUDA_KERNELS_AVAILABLE
        
        # Attention
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, use_cuda=use_cuda_kernels)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # FFN
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps, use_cuda=use_cuda_kernels)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with float32 accumulation for numerical precision."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Store original dtype for output, promote to float32 for computation
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # QKV projections - weights stay fp16 but output is float32
        q = F.linear(hidden_states, self.q_proj.weight.float(), None)
        k = F.linear(hidden_states, self.k_proj.weight.float(), None)
        v = F.linear(hidden_states, self.v_proj.weight.float(), None)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE in float32
        q, k = apply_rope(q, k, cos.float(), sin.float(), position_ids)
        
        # KV cache (store in float32 for precision)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache = k_cache.float()
            v_cache = v_cache.float()
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v)
        
        # Repeat KV for GQA
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Standard attention in float32
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = F.linear(attn_output, self.o_proj.weight.float(), None)
        
        hidden_states = residual + attn_output
        
        # FFN in float32
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # SwiGLU
        gate = F.silu(F.linear(hidden_states, self.gate_proj.weight.float(), None))
        up = F.linear(hidden_states, self.up_proj.weight.float(), None)
        hidden_states = F.linear(gate * up, self.down_proj.weight.float(), None)
        
        hidden_states = residual + hidden_states
        
        # Convert back to original dtype for memory efficiency
        hidden_states = hidden_states.to(original_dtype)
        
        return hidden_states, new_kv_cache


class ZLLMInferenceEngine:
    """
    ZLLM Native Inference Engine with Layer Streaming.
    
    Our own implementation - no external dependencies.
    
    KEY INNOVATION: Layer Streaming
    Instead of loading ALL layers to GPU (OOM on 7B models), we:
    - Calculate optimal GPU layer budget based on available VRAM
    - Keep only N layers on GPU at a time
    - Stream remaining layers from CPU on demand
    - Pin hot layers (12-20) on GPU for speed
    - Prefetch next layers in background
    
    This enables running 7B models on 2-3GB VRAM!
    
    Example:
        engine = ZLLMInferenceEngine("model.gguf")
        output = engine.generate("Hello, world!", max_tokens=100)
        print(output)
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Load a GGUF model for inference with layer streaming.
        
        Args:
            model_path: Path to .gguf file
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype
        
        # Parse GGUF file
        print(f"Loading GGUF: {model_path}")
        self.parser = GGUFParser(model_path)
        self.metadata = self.parser.metadata
        
        # Model architecture params
        self.hidden_size = self.metadata.embedding_length
        self.num_layers = self.metadata.block_count
        self.num_heads = self.metadata.attention_head_count
        self.num_kv_heads = self.metadata.attention_head_count_kv or self.num_heads
        self.vocab_size = self.metadata.vocab_size
        self.intermediate_size = self.metadata.feed_forward_length
        self.max_seq_len = min(self.metadata.context_length, self.config.max_seq_len)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Infer vocab_size from embedding tensor if not in metadata
        if self.vocab_size == 0:
            for name in ["token_embd.weight", "model.embed_tokens.weight", "tok_embeddings.weight"]:
                if name in self.parser.tensors:
                    self.vocab_size = self.parser.tensors[name].shape[1]
                    break
        
        # Fix for models that don't report intermediate_size
        if self.intermediate_size == 0:
            self.intermediate_size = int(self.hidden_size * 2.6875)  # Common ratio
        
        print(f"Model: {self.metadata.architecture}")
        print(f"Layers: {self.num_layers}, Hidden: {self.hidden_size}")
        print(f"Heads: {self.num_heads}, KV Heads: {self.num_kv_heads}")
        print(f"Vocab: {self.vocab_size}, Context: {self.max_seq_len}")
        
        # ============ LAYER STREAMING SETUP ============
        self._setup_layer_streaming()
        
        # Load embeddings and output layers (always in GPU memory - small)
        self.embed_tokens = self._load_embedding()
        self.lm_head = self._load_lm_head()
        self.norm = self._load_final_norm()
        
        # Layer storage
        # GPU cache: layers currently on GPU (limited by budget)
        # CPU cache: layers stored on CPU for streaming
        self._gpu_layers: Dict[int, TransformerLayer] = {}
        self._cpu_layers: Dict[int, TransformerLayer] = {}
        
        # Preload layers based on streaming mode
        self._preload_layers()
        
        # Precompute RoPE cache
        rope_dim = self.metadata.rope_dimension_count or self.head_dim
        self.cos_cache, self.sin_cache = precompute_rope_cache(
            rope_dim,
            self.max_seq_len,
            self.metadata.rope_freq_base,
            device=self.device,
        )
        
        # KV cache
        self.kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
    
    def _setup_layer_streaming(self) -> None:
        """
        Setup layer streaming - THE KEY TO RUNNING 7B ON 2-3GB VRAM.
        
        Calculates optimal GPU layer budget based on:
        1. Available VRAM
        2. Layer size (from model architecture)
        3. Speed mode (fast/balanced/memory)
        4. KV cache reservation
        """
        # Estimate layer size in bytes
        # Each layer has: Q, K, V, O projections + gate, up, down FFN + norms
        attn_size = self.hidden_size * self.hidden_size * 4  # Q, K, V, O
        ffn_size = self.hidden_size * self.intermediate_size * 3  # gate, up, down
        norm_size = self.hidden_size * 2  # two RMSNorms
        
        # In fp16, each param is 2 bytes
        bytes_per_param = 2 if self.dtype == torch.float16 else 4
        self._layer_size_bytes = (attn_size + ffn_size + norm_size) * bytes_per_param
        self._layer_size_mb = self._layer_size_bytes / (1024 * 1024)
        
        # Get available VRAM
        if torch.cuda.is_available() and self.device.type == "cuda":
            free_vram, total_vram = torch.cuda.mem_get_info()
            
            # Reserve for KV cache (about 50% of max cache size)
            kv_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * bytes_per_param
            kv_reserve = kv_per_token * self.max_seq_len // 2
            
            # Reserve for embeddings and lm_head (roughly vocab_size * hidden_size * 2)
            embed_reserve = self.vocab_size * self.hidden_size * bytes_per_param * 2
            
            # Available for layers
            available = free_vram - kv_reserve - embed_reserve
            
            # Speed mode fractions
            mode_fractions = {
                "fast": 0.75,
                "balanced": 0.60,
                "memory": 0.40,
            }
            fraction = mode_fractions.get(self.config.speed_mode, 0.60)
            budget = int(available * fraction)
            
            # Calculate max layers that fit
            self._max_gpu_layers = max(1, budget // self._layer_size_bytes)
            self._max_gpu_layers = min(self._max_gpu_layers, self.num_layers)
        else:
            # CPU mode - all layers on CPU
            self._max_gpu_layers = 0
        
        # Override if user specified
        if self.config.max_gpu_layers is not None:
            self._max_gpu_layers = min(self.config.max_gpu_layers, self.num_layers)
        
        # Calculate hot layers (middle layers are most important)
        # For 28 layers: hot = 10-18 (center)
        hot_start = int(self.num_layers * 0.35)
        hot_end = int(self.num_layers * 0.65)
        self._hot_layers = set(range(hot_start, hot_end + 1))
        
        # Async prefetch executor
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zllm_prefetch")
        self._prefetch_futures: Dict[int, Any] = {}
        
        print(f"Layer Streaming: {self._max_gpu_layers}/{self.num_layers} layers on GPU")
        print(f"Layer size: {self._layer_size_mb:.1f}MB, Hot layers: {hot_start}-{hot_end}")
        print(f"Speed mode: {self.config.speed_mode}")
    
    def _preload_layers(self) -> None:
        """
        Preload layers based on streaming strategy.
        
        Strategy:
        1. If layer_streaming disabled OR all layers fit: load all to GPU
        2. Otherwise: load only hot layers to GPU, rest to CPU
        """
        # ALWAYS load layers to CPU first (key for memory efficiency)
        print(f"Loading {self.num_layers} layers to CPU...")
        for i in range(self.num_layers):
            layer = self._create_and_load_layer(i, device="cpu")
            self._cpu_layers[i] = layer
            if (i + 1) % 5 == 0:
                print(f"  Loaded {i + 1}/{self.num_layers} layers to CPU")
        
        # Now move hot layers to GPU based on budget
        if self._max_gpu_layers > 0:
            print(f"Moving up to {self._max_gpu_layers} hot layers to GPU...")
            loaded_to_gpu = 0
            
            # Prioritize hot (middle) layers
            for i in sorted(self._hot_layers):
                if loaded_to_gpu >= self._max_gpu_layers:
                    break
                layer = self._cpu_layers.pop(i)
                self._gpu_layers[i] = layer.to(self.device)
                loaded_to_gpu += 1
            
            # If we have room, add early layers (0, 1, 2)
            for i in range(3):
                if loaded_to_gpu >= self._max_gpu_layers:
                    break
                if i in self._cpu_layers:
                    layer = self._cpu_layers.pop(i)
                    self._gpu_layers[i] = layer.to(self.device)
                    loaded_to_gpu += 1
        
        print(f"GPU: {len(self._gpu_layers)} layers, CPU: {len(self._cpu_layers)} layers")
    
    def _get_layer(self, layer_idx: int) -> TransformerLayer:
        """
        Get a layer, loading to GPU if needed.
        
        This is where the streaming magic happens:
        1. If layer on GPU: return it
        2. If layer on CPU: move to GPU (evicting if needed)
        3. Start prefetching next layer
        """
        # Already on GPU?
        if layer_idx in self._gpu_layers:
            return self._gpu_layers[layer_idx]
        
        # Need to load from CPU
        if layer_idx not in self._cpu_layers:
            raise RuntimeError(f"Layer {layer_idx} not found on CPU or GPU")
        
        # Check if we need to evict a layer
        if len(self._gpu_layers) >= self._max_gpu_layers:
            self._evict_layer()
        
        # Move layer to GPU
        layer = self._cpu_layers.pop(layer_idx)
        layer = layer.to(self.device)
        self._gpu_layers[layer_idx] = layer
        
        # Start prefetching next layers
        self._prefetch_layers_async(layer_idx)
        
        return layer
    
    def _evict_layer(self) -> None:
        """
        Evict the least important layer from GPU to CPU.
        
        Priority (evict first → evict last):
        1. Low priority (layers 0-2 and last 3)
        2. Normal priority (not in hot zone)
        3. Hot layers (only evict if necessary)
        """
        # Find best candidate to evict
        candidates = []
        for idx in self._gpu_layers.keys():
            if idx in self._hot_layers:
                priority = 2  # Hot - evict last
            elif idx < 3 or idx > self.num_layers - 3:
                priority = 0  # Edge layers - evict first
            else:
                priority = 1  # Normal
            candidates.append((idx, priority))
        
        # Sort by priority (ascending)
        candidates.sort(key=lambda x: x[1])
        
        if candidates:
            evict_idx = candidates[0][0]
            layer = self._gpu_layers.pop(evict_idx)
            self._cpu_layers[evict_idx] = layer.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _prefetch_layers_async(self, current_idx: int) -> None:
        """Prefetch upcoming layers in background thread."""
        if not self.config.layer_streaming:
            return
        
        for offset in range(1, self.config.prefetch_layers + 1):
            next_idx = current_idx + offset
            if next_idx >= self.num_layers:
                break
            if next_idx in self._gpu_layers or next_idx in self._prefetch_futures:
                continue
            if next_idx not in self._cpu_layers:
                continue
            
            # Submit prefetch task
            future = self._prefetch_executor.submit(self._prefetch_layer, next_idx)
            self._prefetch_futures[next_idx] = future
    
    def _prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch a single layer to GPU."""
        try:
            if layer_idx in self._gpu_layers:
                return
            if layer_idx not in self._cpu_layers:
                return
            
            # Check if we need to evict
            if len(self._gpu_layers) >= self._max_gpu_layers:
                self._evict_layer()
            
            layer = self._cpu_layers.pop(layer_idx)
            layer = layer.to(self.device)
            self._gpu_layers[layer_idx] = layer
        except Exception:
            pass  # Ignore prefetch errors
        finally:
            self._prefetch_futures.pop(layer_idx, None)
    
    def _create_and_load_layer(self, layer_idx: int, device: str = "cpu") -> TransformerLayer:
        """Create a TransformerLayer and load weights from GGUF.
        
        Args:
            layer_idx: Layer index
            device: Device to load weights to (default: "cpu" for streaming)
        """
        layer = TransformerLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
        )
        
        # Expected shapes for linear layers
        q_shape = (self.num_heads * self.head_dim, self.hidden_size)
        k_shape = (self.num_kv_heads * self.head_dim, self.hidden_size)
        v_shape = (self.num_kv_heads * self.head_dim, self.hidden_size)
        o_shape = (self.hidden_size, self.num_heads * self.head_dim)
        gate_shape = (self.intermediate_size, self.hidden_size)
        up_shape = (self.intermediate_size, self.hidden_size)
        down_shape = (self.hidden_size, self.intermediate_size)
        
        # Load weights - try different naming conventions
        # ALWAYS load to specified device (CPU for streaming mode)
        prefix = f"model.layers.{layer_idx}"
        
        try:
            # HuggingFace naming
            layer.input_layernorm.weight.data = self._load_tensor(f"{prefix}.input_layernorm.weight", device=device)
            layer.q_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.q_proj.weight", q_shape, device=device)
            layer.k_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.k_proj.weight", k_shape, device=device)
            layer.v_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.v_proj.weight", v_shape, device=device)
            layer.o_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.o_proj.weight", o_shape, device=device)
            layer.post_attention_layernorm.weight.data = self._load_tensor(f"{prefix}.post_attention_layernorm.weight", device=device)
            layer.gate_proj.weight.data = self._load_linear_weight(f"{prefix}.mlp.gate_proj.weight", gate_shape, device=device)
            layer.up_proj.weight.data = self._load_linear_weight(f"{prefix}.mlp.up_proj.weight", up_shape, device=device)
            layer.down_proj.weight.data = self._load_linear_weight(f"{prefix}.mlp.down_proj.weight", down_shape, device=device)
        except KeyError:
            # Try llama.cpp naming (blk.N.*)
            try:
                layer.input_layernorm.weight.data = self._load_tensor(f"blk.{layer_idx}.attn_norm.weight", device=device)
                layer.q_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_q.weight", q_shape, device=device)
                layer.k_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_k.weight", k_shape, device=device)
                layer.v_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_v.weight", v_shape, device=device)
                layer.o_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_output.weight", o_shape, device=device)
                layer.post_attention_layernorm.weight.data = self._load_tensor(f"blk.{layer_idx}.ffn_norm.weight", device=device)
                layer.gate_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.ffn_gate.weight", gate_shape, device=device)
                layer.up_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.ffn_up.weight", up_shape, device=device)
                layer.down_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.ffn_down.weight", down_shape, device=device)
            except KeyError as e:
                raise KeyError(f"Cannot load layer {layer_idx}: {e}")
        
        return layer.to(device).to(self.dtype)

    def _load_tensor(self, name: str, device: str = None) -> torch.Tensor:
        """Load a tensor from GGUF file.
        
        Args:
            name: Tensor name
            device: Target device (default: self.device)
        """
        if name not in self.parser.tensors:
            # Try common name variations
            variations = [
                name,
                name.replace("model.", ""),
                f"model.{name}",
            ]
            for var in variations:
                if var in self.parser.tensors:
                    name = var
                    break
            else:
                raise KeyError(f"Tensor not found: {name}")
        
        target_device = device if device is not None else self.device
        return self.parser.load_tensor(name, target_device).to(self.dtype)
    
    def _load_linear_weight(self, name: str, expected_shape: tuple, device: str = None) -> torch.Tensor:
        """Load a Linear layer weight from GGUF.
        
        GGUF shape is already corrected in load_tensor() to PyTorch convention:
        (out_features, in_features) - no transpose needed.
        """
        weight = self._load_tensor(name, device=device)
        
        # Verify shape matches expected
        if weight.shape != expected_shape:
            print(f"WARNING: {name} shape {weight.shape} != expected {expected_shape}")
        
        return weight
    
    def _load_embedding(self) -> nn.Embedding:
        """Load token embeddings."""
        # Try different naming conventions
        names = [
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight",
            "transformer.wte.weight",
        ]
        
        for name in names:
            if name in self.parser.tensors:
                weight = self._load_tensor(name)
                # GGUF shape is already corrected to (vocab_size, hidden_size)
                embed = nn.Embedding(self.vocab_size, self.hidden_size, device=self.device)
                embed.weight.data = weight
                return embed
        
        raise KeyError(f"Embedding tensor not found. Available: {list(self.parser.tensors.keys())[:5]}")
    
    def _load_lm_head(self) -> nn.Linear:
        """Load output projection (lm_head)."""
        names = [
            "lm_head.weight",
            "model.lm_head.weight",
            "output.weight",
        ]
        
        for name in names:
            if name in self.parser.tensors:
                weight = self._load_tensor(name)
                # GGUF shape is already corrected to (vocab_size, hidden_size)
                lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False, device=self.device)
                lm_head.weight.data = weight
                return lm_head
        
        # If no lm_head, use embedding weights (tied)
        print("Note: Using tied embeddings for lm_head")
        lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False, device=self.device)
        lm_head.weight = self.embed_tokens.weight
        return lm_head
    
    def _load_final_norm(self) -> RMSNorm:
        """Load final layer norm."""
        names = [
            "model.norm.weight",
            "norm.weight",
            "output_norm.weight",
        ]
        
        for name in names:
            if name in self.parser.tensors:
                weight = self._load_tensor(name)
                norm = RMSNorm(self.hidden_size).to(self.device).to(self.dtype)
                norm.weight.data = weight
                return norm
        
        raise KeyError("Final norm tensor not found")
    
    def clear_kv_cache(self) -> None:
        """Clear the KV cache."""
        self.kv_cache = [None] * self.num_layers
    
    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            position_ids: Position IDs (optional)
            use_cache: Whether to use KV cache
        
        Returns:
            Logits tensor (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Default position IDs
        if position_ids is None:
            if use_cache and self.kv_cache[0] is not None:
                past_len = self.kv_cache[0][0].shape[2]
                position_ids = torch.arange(
                    past_len, past_len + seq_len, 
                    device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_len, device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
        
        # Causal mask
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), 
                float("-inf"), 
                device=self.device
            )
            mask = torch.triu(mask, diagonal=1)
            
            if use_cache and self.kv_cache[0] is not None:
                past_len = self.kv_cache[0][0].shape[2]
                mask = torch.cat([
                    torch.zeros((seq_len, past_len), device=self.device),
                    mask
                ], dim=-1)
            
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = None
        
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Debug: check embedding stats
        print(f"Embedding: mean={hidden_states.float().mean():.6f}, std={hidden_states.float().std():.6f}")
        
        # Layers - with streaming support
        for layer_idx in range(self.num_layers):
            # Get layer (streaming: loads from CPU if needed)
            layer = self._get_layer(layer_idx)
            
            kv_cache = self.kv_cache[layer_idx] if use_cache else None
            
            hidden_states, new_kv_cache = layer(
                hidden_states,
                self.cos_cache,
                self.sin_cache,
                position_ids,
                attention_mask=mask,
                kv_cache=kv_cache,
            )
            
            # Debug: check layer stats
            nan_cnt = torch.isnan(hidden_states).sum().item()
            inf_cnt = torch.isinf(hidden_states).sum().item()
            if layer_idx == 0:
                print(f"Layer 0: mean={hidden_states.float().mean():.6f}, std={hidden_states.float().std():.6f}")
            if nan_cnt > 0 or inf_cnt > 0:
                print(f"Layer {layer_idx}: {nan_cnt} nan, {inf_cnt} inf")
            
            if use_cache:
                self.kv_cache[layer_idx] = new_kv_cache
        
        # Final norm and output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Debug: logits stats
        print(f"Logits: mean={logits.float().mean():.6f}, std={logits.float().std():.6f}")
        
        return logits
    
    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        logits = logits[:, -1, :]  # Last position
        
        # Handle inf/nan in logits
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"WARNING: logits contain {nan_count} nan, {inf_count} inf out of {logits.numel()}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        
        logits = logits / temperature
        
        # Clamp to prevent overflow in softmax
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Generate tokens from prompt.
        
        Args:
            prompt_tokens: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_tokens: Token IDs that stop generation
        
        Returns:
            List of generated token IDs
        """
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        stop_tokens = stop_tokens or [self.metadata.eos_token_id]
        
        # Clear cache for new generation
        self.clear_kv_cache()
        
        # Process prompt
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        
        # Prefill (process prompt)
        logits = self.forward(input_ids, use_cache=True)
        
        # Generate
        generated = list(prompt_tokens)
        
        for _ in range(max_new_tokens):
            next_token = self.sample_next_token(
                logits, temperature, top_p, top_k
            )
            
            token_id = next_token.item()
            generated.append(token_id)
            
            if token_id in stop_tokens:
                break
            
            # Decode next token
            logits = self.forward(next_token, use_cache=True)
        
        return generated
    
    def generate_stream(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        **kwargs,
    ) -> Iterator[int]:
        """Stream generated tokens one at a time."""
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        top_k = kwargs.get("top_k", self.config.top_k)
        stop_tokens = kwargs.get("stop_tokens", [self.metadata.eos_token_id])
        
        self.clear_kv_cache()
        
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        logits = self.forward(input_ids, use_cache=True)
        
        for _ in range(max_new_tokens):
            next_token = self.sample_next_token(logits, temperature, top_p, top_k)
            token_id = next_token.item()
            
            yield token_id
            
            if token_id in stop_tokens:
                break
            
            logits = self.forward(next_token, use_cache=True)
    
    def close(self) -> None:
        """Release resources."""
        self.parser.close()
        self.clear_kv_cache()
        
        # Shutdown prefetch executor
        if hasattr(self, '_prefetch_executor') and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)
        
        # Free all layers (GPU and CPU)
        for layer in self._gpu_layers.values():
            del layer
        self._gpu_layers.clear()
        
        for layer in self._cpu_layers.values():
            del layer
        self._cpu_layers.clear()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_engine(
    model_path: str,
    device: str = "auto",
    **kwargs,
) -> ZLLMInferenceEngine:
    """
    Load a GGUF model with our native engine.
    
    Args:
        model_path: Path to .gguf file
        device: Device ('auto', 'cuda', 'cpu')
        **kwargs: Additional config options
    
    Returns:
        ZLLMInferenceEngine instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = InferenceConfig(device=device, **kwargs)
    return ZLLMInferenceEngine(model_path, config)
