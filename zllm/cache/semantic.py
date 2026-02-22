"""
Semantic cache using embedding similarity.

This is a key differentiator for zllm - cache similar prompts, not just exact matches.
"""

import asyncio
import hashlib
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import json

from zllm.cache.base import CacheBackend, CacheEntry


class SemanticCache(CacheBackend):
    """
    Semantic cache that matches similar prompts using embeddings.
    
    Key features:
    - Uses lightweight embedding model (all-MiniLM-L6-v2 by default)
    - Cosine similarity matching
    - Configurable similarity threshold
    - Falls back to exact match if embedding fails
    
    Example:
        "What is the capital of France?" ≈ "Tell me France's capital"
        Both return the same cached response!
    """
    
    # Default embedding model - small and fast
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.92,
        embedding_model: Optional[str] = None,
        persist_path: Optional[Path] = None,
        device: str = "cpu",  # Keep embeddings on CPU to save GPU for LLM
    ):
        """
        Args:
            max_size: Maximum number of cached entries
            similarity_threshold: Minimum similarity to consider a match (0.0-1.0)
            embedding_model: HuggingFace model ID for embeddings
            persist_path: Optional path to persist cache
            device: Device for embedding model ("cpu" recommended)
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model_id = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.persist_path = persist_path
        self.device = device
        
        # Lazy load embedding model
        self._embedding_model = None
        self._model_loaded = False
        
        # Cache storage
        self._entries: Dict[str, CacheEntry] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._keys_order: List[str] = []
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0  # Hits via similarity (not exact match)
        
        self._lock = asyncio.Lock()
        
        # Load persisted cache
        if persist_path and persist_path.exists():
            self._load_from_disk()
    
    def _load_embedding_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self.embedding_model_id,
                device=self.device,
            )
            self._model_loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic cache. "
                "Install with: pip install sentence-transformers"
            )
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        self._load_embedding_model()
        embedding = self._embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Since embeddings are normalized, dot product = cosine similarity
        return float(np.dot(a, b))
    
    def _find_similar(
        self,
        query_embedding: np.ndarray,
    ) -> Optional[Tuple[str, float]]:
        """
        Find the most similar cached entry.
        
        Returns:
            (key, similarity) of best match, or None if no match above threshold
        """
        if not self._embeddings:
            return None
        
        best_key = None
        best_similarity = 0.0
        
        # Compute similarities with all cached embeddings
        for key, embedding in self._embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key
        
        if best_key and best_similarity >= self.similarity_threshold:
            return best_key, best_similarity
        
        return None
    
    def _make_key(self, prompt: str, model_id: str) -> str:
        """Generate a unique key for exact matching."""
        content = f"{model_id}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cached entry by exact key."""
        async with self._lock:
            if key in self._entries:
                self._hits += 1
                entry = self._entries[key]
                entry.hit_count += 1
                return entry
            self._misses += 1
            return None
    
    async def get_semantic(
        self,
        prompt: str,
        model_id: str,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Get a cached entry by semantic similarity.
        
        Args:
            prompt: The query prompt
            model_id: Model identifier
        
        Returns:
            (CacheEntry, similarity_score) if found, None otherwise
        """
        async with self._lock:
            # First try exact match
            exact_key = self._make_key(prompt, model_id)
            if exact_key in self._entries:
                self._hits += 1
                entry = self._entries[exact_key]
                entry.hit_count += 1
                return entry, 1.0
            
            # Try semantic match
            try:
                query_embedding = self._get_embedding(prompt)
                result = self._find_similar(query_embedding)
                
                if result:
                    key, similarity = result
                    entry = self._entries[key]
                    
                    # Only return if same model
                    if entry.model_id == model_id:
                        self._hits += 1
                        self._semantic_hits += 1
                        entry.hit_count += 1
                        return entry, similarity
            except Exception:
                # If embedding fails, just return no match
                pass
            
            self._misses += 1
            return None
    
    async def set(
        self,
        key: str,
        prompt: str,
        response: str,
        model_id: str,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a response in cache."""
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._entries) >= self.max_size and self._keys_order:
                old_key = self._keys_order.pop(0)
                self._entries.pop(old_key, None)
                self._embeddings.pop(old_key, None)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                prompt=prompt,
                response=response,
                model_id=model_id,
                created_at=datetime.now(),
                tokens_used=tokens_used,
                metadata=metadata,
            )
            
            # Compute embedding
            try:
                embedding = self._get_embedding(prompt)
                self._embeddings[key] = embedding
            except Exception:
                # Store without embedding if it fails
                pass
            
            self._entries[key] = entry
            self._keys_order.append(key)
            
            # Persist
            if self.persist_path:
                self._save_to_disk()
    
    async def set_semantic(
        self,
        prompt: str,
        response: str,
        model_id: str,
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a response and return the generated key."""
        key = self._make_key(prompt, model_id)
        await self.set(key, prompt, response, model_id, tokens_used, metadata)
        return key
    
    async def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        async with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._embeddings.pop(key, None)
                if key in self._keys_order:
                    self._keys_order.remove(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._entries.clear()
            self._embeddings.clear()
            self._keys_order.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0
            
            if self.persist_path and self.persist_path.exists():
                self.persist_path.unlink()
    
    async def size(self) -> int:
        """Get the number of entries in cache."""
        return len(self._entries)
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        semantic_rate = self._semantic_hits / self._hits if self._hits > 0 else 0.0
        
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "semantic_hits": self._semantic_hits,
            "hit_rate": hit_rate,
            "semantic_hit_rate": semantic_rate,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model_id,
            "type": "semantic",
        }
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        
        data = {
            "entries": [],
            "embeddings": {},
        }
        
        for key, entry in self._entries.items():
            data["entries"].append({
                "key": entry.key,
                "prompt": entry.prompt,
                "response": entry.response,
                "model_id": entry.model_id,
                "created_at": entry.created_at.isoformat(),
                "tokens_used": entry.tokens_used,
                "hit_count": entry.hit_count,
                "metadata": entry.metadata,
            })
            
            if key in self._embeddings:
                data["embeddings"][key] = self._embeddings[key].tolist()
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f)
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            
            for item in data.get("entries", []):
                entry = CacheEntry(
                    key=item["key"],
                    prompt=item["prompt"],
                    response=item["response"],
                    model_id=item["model_id"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    tokens_used=item.get("tokens_used", 0),
                    hit_count=item.get("hit_count", 0),
                    metadata=item.get("metadata"),
                )
                self._entries[entry.key] = entry
                self._keys_order.append(entry.key)
            
            for key, emb_list in data.get("embeddings", {}).items():
                self._embeddings[key] = np.array(emb_list, dtype=np.float32)
                
        except Exception:
            # If loading fails, start fresh
            pass
