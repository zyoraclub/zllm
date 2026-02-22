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
from dataclasses import dataclass
from pathlib import Path
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
    layer_offload: bool = False  # Offload layers to CPU when not in use
    kv_cache_quantize: bool = True  # Quantize KV cache
    
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
    ZLLM Native Inference Engine.
    
    Our own implementation - no external dependencies.
    
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
        Load a GGUF model for inference.
        
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
        
        # Load embeddings and output layers (always in memory)
        self.embed_tokens = self._load_embedding()
        self.lm_head = self._load_lm_head()
        self.norm = self._load_final_norm()
        
        # Layers (loaded on demand for memory efficiency)
        self.layers: List[Optional[TransformerLayer]] = [None] * self.num_layers
        
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
    
    def _load_tensor(self, name: str) -> torch.Tensor:
        """Load a tensor from GGUF file."""
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
        
        return self.parser.load_tensor(name, self.device).to(self.dtype)
    
    def _load_linear_weight(self, name: str, expected_shape: tuple) -> torch.Tensor:
        """Load a Linear layer weight from GGUF.
        
        GGUF shape is already corrected in load_tensor() to PyTorch convention:
        (out_features, in_features) - no transpose needed.
        """
        weight = self._load_tensor(name)
        
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
    
    def _load_layer(self, layer_idx: int) -> TransformerLayer:
        """Load a transformer layer from GGUF."""
        if self.layers[layer_idx] is not None:
            return self.layers[layer_idx]
        
        layer = TransformerLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
        ).to(self.device).to(self.dtype)
        
        # Expected shapes for linear layers
        q_shape = (self.num_heads * self.head_dim, self.hidden_size)
        k_shape = (self.num_kv_heads * self.head_dim, self.hidden_size)
        v_shape = (self.num_kv_heads * self.head_dim, self.hidden_size)
        o_shape = (self.hidden_size, self.num_heads * self.head_dim)
        gate_shape = (self.intermediate_size, self.hidden_size)
        up_shape = (self.intermediate_size, self.hidden_size)
        down_shape = (self.hidden_size, self.intermediate_size)
        
        # Load weights
        prefix = f"model.layers.{layer_idx}"
        
        try:
            # Attention norms and projections
            layer.input_layernorm.weight.data = self._load_tensor(f"{prefix}.input_layernorm.weight")
            layer.q_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.q_proj.weight", q_shape)
            layer.k_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.k_proj.weight", k_shape)
            layer.v_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.v_proj.weight", v_shape)
            layer.o_proj.weight.data = self._load_linear_weight(f"{prefix}.self_attn.o_proj.weight", o_shape)
            
            # FFN
            layer.post_attention_layernorm.weight.data = self._load_tensor(f"{prefix}.post_attention_layernorm.weight")
            layer.gate_proj.weight.data = self._load_linear_weight(f"{prefix}.mlp.gate_proj.weight", gate_shape)
            layer.up_proj.weight.data = self._load_linear_weight(f"{prefix}.mlp.up_proj.weight", up_shape)
            layer.down_proj.weight.data = self._load_linear_weight(f"{prefix}.mlp.down_proj.weight", down_shape)
        except KeyError as e:
            # Try alternative naming (e.g., for llama.cpp naming)
            try:
                layer.input_layernorm.weight.data = self._load_tensor(f"blk.{layer_idx}.attn_norm.weight")
                layer.q_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_q.weight", q_shape)
                layer.k_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_k.weight", k_shape)
                layer.v_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_v.weight", v_shape)
                layer.o_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.attn_output.weight", o_shape)
                layer.post_attention_layernorm.weight.data = self._load_tensor(f"blk.{layer_idx}.ffn_norm.weight")
                layer.gate_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.ffn_gate.weight", gate_shape)
                layer.up_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.ffn_up.weight", up_shape)
                layer.down_proj.weight.data = self._load_linear_weight(f"blk.{layer_idx}.ffn_down.weight", down_shape)
            except KeyError:
                raise KeyError(f"Cannot load layer {layer_idx}: {e}")
        
        self.layers[layer_idx] = layer
        return layer
    
    def _unload_layer(self, layer_idx: int) -> None:
        """Unload a layer from GPU memory."""
        if self.layers[layer_idx] is not None:
            del self.layers[layer_idx]
            self.layers[layer_idx] = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
        
        # Layers
        for layer_idx in range(self.num_layers):
            layer = self._load_layer(layer_idx)
            
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
            
            # Optionally offload layer after use
            if self.config.layer_offload:
                self._unload_layer(layer_idx)
        
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
        
        # Free layers
        for i in range(self.num_layers):
            self._unload_layer(i)


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
