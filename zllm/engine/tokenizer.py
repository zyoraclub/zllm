"""
Simple tokenizer for GGUF models.

Reads tokenizer vocabulary from GGUF metadata.
Supports BPE and SentencePiece tokenization.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re


class SimpleTokenizer:
    """
    Simple tokenizer that uses vocabulary from GGUF file.
    
    Supports basic BPE-style tokenization.
    """
    
    def __init__(
        self,
        vocab: List[str],
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: List of tokens (index = token_id)
            merges: Optional BPE merges
            special_tokens: Special token mapping
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        # Build token -> ID mapping
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        
        # BPE merges
        self.merges = merges or []
        self.bpe_ranks = {merge: i for i, merge in enumerate(self.merges)}
        
        # Special tokens
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        self.special_tokens = special_tokens or {}
        
        # Cache for BPE
        self._cache: Dict[str, List[str]] = {}
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token
        
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        # Simple word-level + subword tokenization
        # Split by whitespace and punctuation
        words = self._split_text(text)
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip BOS/EOS tokens
        
        Returns:
            Decoded text
        """
        special = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special:
                continue
            
            if 0 <= token_id < len(self.vocab):
                token = self.vocab[token_id]
                tokens.append(token)
        
        # Join tokens
        text = "".join(tokens)
        
        # Clean up common tokenizer artifacts
        text = text.replace("Ġ", " ")  # GPT-2 style space
        text = text.replace("▁", " ")  # SentencePiece style space
        text = text.replace("<0x0A>", "\n")  # Newline
        
        return text.strip()
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into words/subwords for tokenization."""
        # Simple split by whitespace, keeping punctuation separate
        pattern = r"(\s+|[.,!?;:\"'()\[\]{}])"
        parts = re.split(pattern, text)
        return [p for p in parts if p]
    
    def _tokenize_word(self, word: str) -> List[int]:
        """Tokenize a single word using BPE or fallback."""
        # Check if whole word is in vocab
        if word in self.token_to_id:
            return [self.token_to_id[word]]
        
        # Try with space prefix (common in BPE)
        space_word = "Ġ" + word
        if space_word in self.token_to_id:
            return [self.token_to_id[space_word]]
        
        space_word = "▁" + word
        if space_word in self.token_to_id:
            return [self.token_to_id[space_word]]
        
        # BPE tokenization
        if self.merges:
            return self._bpe_tokenize(word)
        
        # Fallback: character-level
        return self._char_tokenize(word)
    
    def _bpe_tokenize(self, word: str) -> List[int]:
        """Apply BPE tokenization."""
        if word in self._cache:
            tokens = self._cache[word]
        else:
            tokens = list(word)
            
            while len(tokens) > 1:
                # Find best merge
                best_pair = None
                best_rank = float("inf")
                
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.bpe_ranks:
                        rank = self.bpe_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                
                if best_pair is None:
                    break
                
                # Apply merge
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            self._cache[word] = tokens
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Unknown token - use character fallback
                ids.extend(self._char_tokenize(token))
        
        return ids
    
    def _char_tokenize(self, word: str) -> List[int]:
        """Character-level tokenization as fallback."""
        ids = []
        for char in word:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                # Try byte representation
                for byte in char.encode("utf-8"):
                    byte_token = f"<0x{byte:02X}>"
                    if byte_token in self.token_to_id:
                        ids.append(self.token_to_id[byte_token])
        return ids


def load_tokenizer_from_gguf(parser) -> SimpleTokenizer:
    """
    Load tokenizer from GGUF parser.
    
    Args:
        parser: GGUFParser instance
    
    Returns:
        SimpleTokenizer instance
    """
    metadata = parser.metadata.raw
    
    # Get vocabulary
    vocab = metadata.get("tokenizer.ggml.tokens", [])
    if not vocab:
        raise ValueError("No tokenizer vocabulary found in GGUF file")
    
    # Get merges (if available)
    merges = None
    merge_data = metadata.get("tokenizer.ggml.merges", [])
    if merge_data:
        merges = []
        for merge in merge_data:
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    
    # Get special tokens
    bos_id = metadata.get("tokenizer.ggml.bos_token_id", 1)
    eos_id = metadata.get("tokenizer.ggml.eos_token_id", 2)
    pad_id = metadata.get("tokenizer.ggml.padding_token_id", 0)
    
    return SimpleTokenizer(
        vocab=vocab,
        merges=merges,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )


class ChatTemplate:
    """
    Apply chat template to messages.
    
    Supports common formats: ChatML, Llama, Alpaca, etc.
    """
    
    TEMPLATES = {
        "chatml": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
        },
        "llama": {
            "system": "<<SYS>>\n{content}\n<</SYS>>\n\n",
            "user": "[INST] {content} [/INST]",
            "assistant": " {content} ",
            "assistant_start": "",
        },
        "alpaca": {
            "system": "### Instruction:\n{content}\n\n",
            "user": "### Input:\n{content}\n\n",
            "assistant": "### Response:\n{content}\n\n",
            "assistant_start": "### Response:\n",
        },
        "vicuna": {
            "system": "{content}\n\n",
            "user": "USER: {content}\n",
            "assistant": "ASSISTANT: {content}\n",
            "assistant_start": "ASSISTANT:",
        },
    }
    
    def __init__(self, template_name: str = "chatml"):
        """
        Initialize chat template.
        
        Args:
            template_name: Template name (chatml, llama, alpaca, vicuna)
        """
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        
        self.template = self.TEMPLATES[template_name]
    
    def apply(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply template to messages.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            add_generation_prompt: Add assistant start token at end
        
        Returns:
            Formatted prompt string
        """
        prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role in self.template:
                prompt += self.template[role].format(content=content)
        
        if add_generation_prompt:
            prompt += self.template.get("assistant_start", "")
        
        return prompt
    
    @classmethod
    def detect_template(cls, model_name: str) -> str:
        """Auto-detect template from model name."""
        model_lower = model_name.lower()
        
        if "qwen" in model_lower or "chatml" in model_lower:
            return "chatml"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "llama"
        elif "alpaca" in model_lower:
            return "alpaca"
        elif "vicuna" in model_lower:
            return "vicuna"
        else:
            return "chatml"  # Default
