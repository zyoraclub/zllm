"""
Base cache interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CacheEntry:
    """A cached response entry."""
    key: str
    prompt: str
    response: str
    model_id: str
    created_at: datetime
    tokens_used: int = 0
    hit_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cached entry by key."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get the number of entries in cache."""
        pass
    
    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass
