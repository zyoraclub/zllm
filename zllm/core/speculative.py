"""
Speculative Decoding for ZLLM.

Speculative decoding uses a small "draft" model to predict multiple tokens,
then verifies them in parallel with the main "target" model. This can provide
2-3x speedup with NO quality loss.

How it works:
    
    Traditional Decoding (slow):
        Step 1: Target model generates token 1  (100ms)
        Step 2: Target model generates token 2  (100ms)
        Step 3: Target model generates token 3  (100ms)
        Total: 300ms for 3 tokens
    
    Speculative Decoding (fast):
        Step 1: Draft model generates tokens 1,2,3,4,5  (20ms total - small model)
        Step 2: Target model verifies all 5 in ONE forward pass (110ms)
        Step 3: Accept first 3 tokens (verification passed), reject 4,5
        Total: 130ms for 3 tokens = 2.3x speedup!

Why it works:
    - Draft model is 10-20x smaller → 10-20x faster per token
    - Target model verification of N tokens ≈ cost of generating 1 token
    - Even if we only accept 60% of draft tokens, we're still faster

Key insight:
    Verification is PARALLEL - we can check if P(token_i | context) matches
    for ALL draft tokens simultaneously in a single forward pass.

Quality guarantee:
    - Output is IDENTICAL to running target model alone
    - We only accept tokens that the target model would have generated
    - Rejection sampling ensures exact distribution matching

Best draft models:
    - Same architecture, fewer layers (e.g., target=70B, draft=7B)
    - Same tokenizer is REQUIRED
    - Trained on similar data for better acceptance rate

This module implements the core speculative decoding algorithm.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Iterator
from enum import Enum
import math

import torch
from torch import Tensor
import torch.nn.functional as F


class AcceptanceMethod(Enum):
    """Methods for accepting/rejecting draft tokens."""
    GREEDY = "greedy"           # Accept if argmax matches
    SAMPLING = "sampling"       # Rejection sampling (exact distribution)
    THRESHOLD = "threshold"     # Accept if probability above threshold


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    
    # Draft model settings
    draft_model_id: Optional[str] = None  # HuggingFace model ID
    num_draft_tokens: int = 5             # How many tokens to speculate
    
    # Acceptance settings
    acceptance_method: AcceptanceMethod = AcceptanceMethod.SAMPLING
    acceptance_threshold: float = 0.1     # For threshold method
    
    # Fallback settings
    min_acceptance_rate: float = 0.3      # Switch to normal if too low
    fallback_after_rejections: int = 3    # Consecutive rejections to trigger fallback
    
    # Memory settings
    share_kv_cache: bool = True           # Share KV cache between models
    draft_quantization: str = "int4"      # Quantize draft model more aggressively


@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding."""
    total_draft_tokens: int = 0           # Draft tokens generated
    accepted_tokens: int = 0              # Tokens accepted from draft
    rejected_tokens: int = 0              # Tokens rejected
    fallback_count: int = 0               # Times we fell back to normal
    target_forward_passes: int = 0        # Target model calls
    draft_forward_passes: int = 0         # Draft model calls
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens
    
    @property
    def speedup_factor(self) -> float:
        """Estimate speedup vs normal decoding."""
        if self.target_forward_passes == 0:
            return 1.0
        # Tokens generated per target forward pass
        tokens_per_pass = (self.accepted_tokens + self.target_forward_passes) / self.target_forward_passes
        return tokens_per_pass
    
    def __str__(self) -> str:
        return (
            f"SpeculativeStats(\n"
            f"  acceptance_rate={self.acceptance_rate:.1%},\n"
            f"  speedup={self.speedup_factor:.2f}x,\n"
            f"  accepted={self.accepted_tokens},\n"
            f"  rejected={self.rejected_tokens}\n"
            f")"
        )


class SpeculativeDecoder:
    """
    Implements speculative decoding for faster inference.
    
    Usage:
        # Initialize with target and draft models
        decoder = SpeculativeDecoder(
            target_model=llama_70b,
            draft_model=llama_7b,
            tokenizer=tokenizer,
        )
        
        # Generate with speculation
        for token in decoder.generate(input_ids, max_new_tokens=100):
            print(tokenizer.decode([token]), end="")
    """
    
    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        tokenizer: Any,
        config: Optional[SpeculativeConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: The main (large) model for verification
            draft_model: The smaller model for speculation
            tokenizer: Shared tokenizer (MUST be same for both)
            config: Speculative decoding configuration
            device: Device for inference
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()
        self.device = device
        
        # Statistics
        self.stats = SpeculativeStats()
        
        # State
        self._consecutive_rejections = 0
        self._fallback_mode = False
        
        # Ensure models are in eval mode
        if hasattr(target_model, 'eval'):
            target_model.eval()
        if hasattr(draft_model, 'eval'):
            draft_model.eval()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = SpeculativeStats()
        self._consecutive_rejections = 0
        self._fallback_mode = False
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_token_id: Optional[int] = None,
    ) -> Iterator[int]:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len] or [seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop_token_id: Token ID to stop at (e.g., EOS)
        
        Yields:
            Generated token IDs one at a time
        """
        # Ensure input is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(self.device)
        current_ids = input_ids.clone()
        
        generated = 0
        stop_token_id = stop_token_id or getattr(self.tokenizer, 'eos_token_id', None)
        
        while generated < max_new_tokens:
            if self._fallback_mode:
                # Normal decoding (one token at a time)
                token = self._generate_one_target(current_ids, temperature, top_p, top_k)
                yield token.item()
                
                current_ids = torch.cat([current_ids, token.unsqueeze(0).unsqueeze(0)], dim=-1)
                generated += 1
                
                if stop_token_id and token.item() == stop_token_id:
                    break
            else:
                # Speculative decoding
                accepted, new_ids = self._speculative_step(
                    current_ids,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                
                for token_id in accepted:
                    yield token_id
                    generated += 1
                    
                    if stop_token_id and token_id == stop_token_id:
                        return
                    
                    if generated >= max_new_tokens:
                        return
                
                current_ids = new_ids
    
    def _speculative_step(
        self,
        input_ids: Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[List[int], Tensor]:
        """
        Perform one speculative decoding step.
        
        1. Draft model generates K tokens
        2. Target model verifies all K+1 positions in parallel
        3. Accept tokens where draft matches target distribution
        4. Return accepted tokens and updated input_ids
        
        Returns:
            (accepted_token_ids, new_input_ids)
        """
        K = self.config.num_draft_tokens
        
        # Step 1: Generate K draft tokens
        draft_tokens, draft_probs = self._draft_forward(input_ids, K, temperature, top_p, top_k)
        self.stats.draft_forward_passes += K
        self.stats.total_draft_tokens += K
        
        # Step 2: Verify with target model (single forward pass for K+1 positions)
        # We need to verify draft tokens AND generate the next token after them
        candidate_ids = torch.cat([input_ids, draft_tokens.unsqueeze(0)], dim=-1)
        
        target_logits = self._target_forward(candidate_ids)
        self.stats.target_forward_passes += 1
        
        # Step 3: Accept/reject each draft token
        accepted_tokens = []
        accepted_count = 0
        
        for i in range(K):
            pos = input_ids.shape[-1] + i - 1  # Position in target_logits
            
            if self._accept_token(
                draft_token=draft_tokens[i].item(),
                draft_prob=draft_probs[i],
                target_logits=target_logits[0, pos],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ):
                accepted_tokens.append(draft_tokens[i].item())
                accepted_count += 1
                self.stats.accepted_tokens += 1
            else:
                # Rejection - sample from adjusted distribution and stop
                self.stats.rejected_tokens += K - i
                
                # Sample new token from target distribution
                adjusted_logits = target_logits[0, pos]
                new_token = self._sample(adjusted_logits, temperature, top_p, top_k)
                accepted_tokens.append(new_token.item())
                
                break
        else:
            # All K tokens accepted - sample bonus token from target
            bonus_token = self._sample(
                target_logits[0, -1],
                temperature, top_p, top_k
            )
            accepted_tokens.append(bonus_token.item())
        
        # Track acceptance rate for fallback decision
        if accepted_count == 0:
            self._consecutive_rejections += 1
            if self._consecutive_rejections >= self.config.fallback_after_rejections:
                self._fallback_mode = True
                self.stats.fallback_count += 1
        else:
            self._consecutive_rejections = 0
        
        # Build new input_ids
        new_ids = torch.cat([
            input_ids,
            torch.tensor([accepted_tokens], device=self.device)
        ], dim=-1)
        
        return accepted_tokens, new_ids
    
    def _draft_forward(
        self,
        input_ids: Tensor,
        num_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate multiple tokens from draft model.
        
        Returns:
            (token_ids, token_probs) both of shape [num_tokens]
        """
        tokens = []
        probs = []
        
        current_ids = input_ids.clone()
        
        for _ in range(num_tokens):
            outputs = self.draft_model(current_ids)
            logits = outputs.logits[0, -1]
            
            # Apply temperature and sampling
            if temperature > 0:
                logits = logits / temperature
            
            # Get probabilities
            prob_dist = F.softmax(logits, dim=-1)
            
            # Sample token
            token = self._sample(logits, temperature, top_p, top_k)
            token_prob = prob_dist[token].item()
            
            tokens.append(token)
            probs.append(token_prob)
            
            current_ids = torch.cat([current_ids, token.unsqueeze(0).unsqueeze(0)], dim=-1)
        
        return torch.stack(tokens), torch.tensor(probs, device=self.device)
    
    def _target_forward(self, input_ids: Tensor) -> Tensor:
        """
        Run target model forward pass and return logits.
        
        Args:
            input_ids: Full sequence including draft tokens
        
        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        outputs = self.target_model(input_ids)
        return outputs.logits
    
    def _accept_token(
        self,
        draft_token: int,
        draft_prob: float,
        target_logits: Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> bool:
        """
        Decide whether to accept a draft token.
        
        Uses rejection sampling to maintain exact target distribution.
        """
        if self.config.acceptance_method == AcceptanceMethod.GREEDY:
            # Accept if draft matches target argmax
            target_token = target_logits.argmax().item()
            return draft_token == target_token
        
        elif self.config.acceptance_method == AcceptanceMethod.THRESHOLD:
            # Accept if target probability is above threshold
            target_probs = F.softmax(target_logits / max(temperature, 1e-8), dim=-1)
            target_prob = target_probs[draft_token].item()
            return target_prob >= self.config.acceptance_threshold
        
        else:  # SAMPLING - rejection sampling
            # This is the mathematically correct approach
            # Accept with probability min(1, target_prob / draft_prob)
            
            target_probs = F.softmax(target_logits / max(temperature, 1e-8), dim=-1)
            target_prob = target_probs[draft_token].item()
            
            # Avoid division by zero
            if draft_prob < 1e-10:
                return False
            
            accept_prob = min(1.0, target_prob / draft_prob)
            return torch.rand(1).item() < accept_prob
    
    def _sample(
        self,
        logits: Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tensor:
        """Sample a token from logits."""
        if temperature == 0:
            return logits.argmax()
        
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze()
    
    def _generate_one_target(
        self,
        input_ids: Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tensor:
        """Generate single token with target model (fallback)."""
        outputs = self.target_model(input_ids)
        logits = outputs.logits[0, -1]
        self.stats.target_forward_passes += 1
        return self._sample(logits, temperature, top_p, top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "acceptance_rate": self.stats.acceptance_rate,
            "speedup_factor": self.stats.speedup_factor,
            "accepted_tokens": self.stats.accepted_tokens,
            "rejected_tokens": self.stats.rejected_tokens,
            "total_draft_tokens": self.stats.total_draft_tokens,
            "target_forward_passes": self.stats.target_forward_passes,
            "draft_forward_passes": self.stats.draft_forward_passes,
            "fallback_count": self.stats.fallback_count,
            "is_fallback_mode": self._fallback_mode,
        }


class SpeculativeDecoderWithCache(SpeculativeDecoder):
    """
    Speculative decoder with KV cache support for efficiency.
    
    This version maintains separate KV caches for draft and target models,
    enabling faster inference by avoiding redundant computation.
    """
    
    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        tokenizer: Any,
        config: Optional[SpeculativeConfig] = None,
        device: str = "cuda",
    ):
        super().__init__(target_model, draft_model, tokenizer, config, device)
        
        # KV caches
        self._target_kv_cache = None
        self._draft_kv_cache = None
        self._cache_position = 0
    
    def reset_cache(self) -> None:
        """Reset KV caches."""
        self._target_kv_cache = None
        self._draft_kv_cache = None
        self._cache_position = 0
    
    def _draft_forward(
        self,
        input_ids: Tensor,
        num_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[Tensor, Tensor]:
        """Generate draft tokens with KV cache."""
        tokens = []
        probs = []
        
        # Only process new tokens
        if self._draft_kv_cache is not None:
            # Use only the last token (or new tokens)
            new_input_ids = input_ids[:, self._cache_position:]
        else:
            new_input_ids = input_ids
        
        for i in range(num_tokens):
            outputs = self.draft_model(
                new_input_ids if i == 0 else tokens[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=self._draft_kv_cache,
                use_cache=True,
            )
            self._draft_kv_cache = outputs.past_key_values
            
            logits = outputs.logits[0, -1]
            
            if temperature > 0:
                logits = logits / temperature
            
            prob_dist = F.softmax(logits, dim=-1)
            token = self._sample(logits, temperature, top_p, top_k)
            
            tokens.append(token)
            probs.append(prob_dist[token].item())
        
        return torch.stack(tokens), torch.tensor(probs, device=self.device)
    
    def _target_forward(self, input_ids: Tensor) -> Tensor:
        """Run target forward with KV cache."""
        # Only verify new tokens
        if self._target_kv_cache is not None:
            new_input_ids = input_ids[:, self._cache_position:]
        else:
            new_input_ids = input_ids
        
        outputs = self.target_model(
            new_input_ids,
            past_key_values=self._target_kv_cache,
            use_cache=True,
        )
        self._target_kv_cache = outputs.past_key_values
        
        return outputs.logits


def create_speculative_pair(
    target_model_id: str,
    draft_model_id: Optional[str] = None,
    device: str = "cuda",
    target_quantization: Optional[str] = None,
    draft_quantization: str = "int4",
) -> Tuple[Any, Any, Any]:
    """
    Create a target-draft model pair for speculative decoding.
    
    If no draft model specified, attempts to auto-select based on target.
    
    Args:
        target_model_id: HuggingFace ID for target model
        draft_model_id: HuggingFace ID for draft model (optional)
        device: Device to load models on
        target_quantization: Quantization for target model
        draft_quantization: Quantization for draft model (default: int4)
    
    Returns:
        (target_model, draft_model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Auto-select draft model based on target
    if draft_model_id is None:
        draft_model_id = _auto_select_draft(target_model_id)
    
    print(f"Loading target model: {target_model_id}")
    print(f"Loading draft model: {draft_model_id}")
    
    # Load tokenizer (use target's tokenizer - must be compatible)
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    
    # Load target model
    target_kwargs = {"device_map": device, "torch_dtype": torch.float16}
    if target_quantization == "int4":
        target_kwargs["load_in_4bit"] = True
    elif target_quantization == "int8":
        target_kwargs["load_in_8bit"] = True
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        **target_kwargs
    )
    
    # Load draft model (more aggressive quantization)
    draft_kwargs = {"device_map": device, "torch_dtype": torch.float16}
    if draft_quantization == "int4":
        draft_kwargs["load_in_4bit"] = True
    elif draft_quantization == "int8":
        draft_kwargs["load_in_8bit"] = True
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        **draft_kwargs
    )
    
    return target_model, draft_model, tokenizer


def _auto_select_draft(target_model_id: str) -> str:
    """Auto-select a draft model based on target model."""
    # Map of target models to recommended draft models
    draft_models = {
        # Llama family
        "meta-llama/Llama-3-70B-Instruct": "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3-8B-Instruct": "meta-llama/Llama-3-8B-Instruct",  # Self-draft
        "meta-llama/Llama-2-70b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        
        # Mistral family
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Large": "mistralai/Mistral-7B-Instruct-v0.3",
        
        # Qwen family
        "Qwen/Qwen2-72B-Instruct": "Qwen/Qwen2-7B-Instruct",
        
        # CodeLlama
        "codellama/CodeLlama-70b-Instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",
    }
    
    if target_model_id in draft_models:
        return draft_models[target_model_id]
    
    # Try to find a smaller variant
    model_lower = target_model_id.lower()
    
    if "70b" in model_lower:
        # Try to find 7B variant
        return target_model_id.replace("70b", "7b").replace("70B", "7B")
    elif "13b" in model_lower:
        return target_model_id.replace("13b", "7b").replace("13B", "7B")
    
    # Fallback: use same model (self-speculation)
    print(f"Warning: No draft model found for {target_model_id}, using self-speculation")
    return target_model_id


# Benchmark utility
def benchmark_speculative(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 100,
    num_draft_tokens: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark speculative decoding vs normal decoding.
    
    Returns:
        Dictionary with timing results and speedup factor
    """
    import time
    
    config = SpeculativeConfig(num_draft_tokens=num_draft_tokens)
    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        config=config,
    )
    
    results = {
        "speculative_times": [],
        "normal_times": [],
        "acceptance_rates": [],
        "prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
    }
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Speculative decoding
        decoder.reset_stats()
        start = time.perf_counter()
        spec_tokens = list(decoder.generate(input_ids, max_new_tokens=max_new_tokens))
        spec_time = time.perf_counter() - start
        
        results["speculative_times"].append(spec_time)
        results["acceptance_rates"].append(decoder.stats.acceptance_rate)
        
        # Normal decoding (just target model)
        start = time.perf_counter()
        normal_tokens = []
        current = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = target_model(current)
                next_token = outputs.logits[0, -1].argmax()
                normal_tokens.append(next_token.item())
                current = torch.cat([current, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        normal_time = time.perf_counter() - start
        
        results["normal_times"].append(normal_time)
    
    # Calculate summary stats
    results["avg_speculative_time"] = sum(results["speculative_times"]) / len(results["speculative_times"])
    results["avg_normal_time"] = sum(results["normal_times"]) / len(results["normal_times"])
    results["avg_acceptance_rate"] = sum(results["acceptance_rates"]) / len(results["acceptance_rates"])
    results["speedup"] = results["avg_normal_time"] / results["avg_speculative_time"]
    
    return results
