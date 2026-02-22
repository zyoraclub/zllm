"""
Generation configuration and text generation logic.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator, Callable
import torch


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Stop conditions
    stop_strings: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    
    # Streaming
    stream: bool = False
    
    # System prompt
    system_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transformers generate()."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }


@dataclass
class GenerationOutput:
    """Output from text generation."""
    text: str
    tokens_generated: int
    finish_reason: str  # "stop", "length", "error"
    prompt_tokens: int = 0
    total_tokens: int = 0
    time_seconds: float = 0.0
    
    @property
    def tokens_per_second(self) -> float:
        if self.time_seconds > 0:
            return self.tokens_generated / self.time_seconds
        return 0.0


class TextGenerator:
    """
    Text generation with streaming support.
    
    Handles:
    - Chat formatting
    - Streaming generation
    - Stop string detection
    - Token counting
    """
    
    # Common chat templates
    CHAT_TEMPLATES = {
        "llama3": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "mistral": (
            "<s>[INST] {system}\n\n{user} [/INST]"
        ),
        "chatml": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "default": "{system}\n\nUser: {user}\n\nAssistant:",
    }
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Detect chat template
        self.chat_template = self._detect_template()
    
    def _detect_template(self) -> str:
        """Detect the appropriate chat template for the model."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return "native"  # Use tokenizer's built-in template
        
        # Detect from model name or config
        model_name = getattr(self.model.config, "_name_or_path", "").lower()
        
        if "llama-3" in model_name or "llama3" in model_name:
            return "llama3"
        elif "mistral" in model_name or "mixtral" in model_name:
            return "mistral"
        elif "qwen" in model_name or "phi" in model_name:
            return "chatml"
        
        return "default"
    
    def format_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Format a prompt for the model.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            history: Optional conversation history
        
        Returns:
            Formatted prompt string
        """
        system = system_prompt or "You are a helpful assistant."
        
        if self.chat_template == "native":
            # Use tokenizer's apply_chat_template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system})
            
            if history:
                messages.extend(history)
            
            messages.append({"role": "user", "content": user_message})
            
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Use our templates
        template = self.CHAT_TEMPLATES.get(self.chat_template, self.CHAT_TEMPLATES["default"])
        return template.format(system=system, user=user_message)
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The formatted prompt
            config: Generation configuration
        
        Returns:
            GenerationOutput with the generated text
        """
        import time
        
        config = config or GenerationConfig()
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **config.to_dict(),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_ids = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop strings
        for stop in config.stop_strings:
            if stop in text:
                text = text[:text.index(stop)]
        
        elapsed = time.time() - start_time
        tokens_generated = len(generated_ids)
        
        finish_reason = "stop"
        if tokens_generated >= config.max_new_tokens:
            finish_reason = "length"
        
        return GenerationOutput(
            text=text.strip(),
            tokens_generated=tokens_generated,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens + tokens_generated,
            time_seconds=elapsed,
        )
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.
        
        Args:
            prompt: The formatted prompt
            config: Generation configuration
            callback: Optional callback for each token
        
        Yields:
            Generated text tokens
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        config = config or GenerationConfig()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Generate in background thread
        generation_kwargs = {
            **inputs,
            **config.to_dict(),
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        generated_text = ""
        for token in streamer:
            generated_text += token
            
            # Check stop strings
            stop_found = False
            for stop in config.stop_strings:
                if stop in generated_text:
                    # Yield up to stop string and exit
                    final = generated_text[:generated_text.index(stop)]
                    if final[len(generated_text) - len(token):]:
                        yield final[len(generated_text) - len(token):]
                    stop_found = True
                    break
            
            if stop_found:
                break
            
            if callback:
                callback(token)
            
            yield token
        
        thread.join()
