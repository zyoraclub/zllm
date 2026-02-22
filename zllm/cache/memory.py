"""
In-memory LRU cache with optional persistence.
"""

import asyncio
import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from zllm.cache.base import CacheBackend, CacheEntry


class MemoryCache(CacheBackend):
    """
    In-memory LRU cache for prompt responses.
    
    Features:
    - LRU eviction
    - Optional persistence to disk
    - Exact match caching
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        persist_path: Optional[Path] = None,
    ):
        """
        Args:
            max_size: Maximum number of entries
            persist_path: Optional path to persist cache to disk
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
        
        # Load persisted cache if available
        if persist_path and persist_path.exists():
            self._load_from_disk()
    
    def _make_key(self, prompt: str, model_id: str) -> str:
        """Generate a cache key from prompt and model."""
        content = f"{model_id}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cached entry by key."""
        async with self._lock:
            if key in self._cache:
                self._hits += 1
                entry = self._cache[key]
                entry.hit_count += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return entry
            self._misses += 1
            return None
    
    async def get_by_prompt(
        self,
        prompt: str,
        model_id: str,
    ) -> Optional[CacheEntry]:
        """Get a cached entry by prompt and model."""
        key = self._make_key(prompt, model_id)
        return await self.get(key)
    
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
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            entry = CacheEntry(
                key=key,
                prompt=prompt,
                response=response,
                model_id=model_id,
                created_at=datetime.now(),
                tokens_used=tokens_used,
                metadata=metadata,
            )
            
            self._cache[key] = entry
            
            # Persist if configured
            if self.persist_path:
                self._save_to_disk()
    
    async def set_by_prompt(
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
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            
            if self.persist_path and self.persist_path.exists():
                self.persist_path.unlink()
    
    async def size(self) -> int:
        """Get the number of entries in cache."""
        return len(self._cache)
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "type": "memory",
        }
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        
        data = []
        for entry in self._cache.values():
            data.append({
                "key": entry.key,
                "prompt": entry.prompt,
                "response": entry.response,
                "model_id": entry.model_id,
                "created_at": entry.created_at.isoformat(),
                "tokens_used": entry.tokens_used,
                "hit_count": entry.hit_count,
                "metadata": entry.metadata,
            })
        
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
            
            for item in data:
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
                self._cache[entry.key] = entry
        except Exception:
            # If loading fails, start fresh
            pass
