"""
Abstract base class for cache backends.

This module defines the interface that all cache implementations must follow,
ensuring consistent behavior across different caching strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Type

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class CacheBackend(ABC):
    """Abstract base class for cache backends.
    
    All cache implementations must inherit from this class and implement
    the required methods. This ensures a consistent interface across
    different caching strategies (LRU, Redis, Disk, etc.).
    
    The cache is designed specifically for Pydantic models and includes
    schema-aware invalidation to prevent stale data when models change.
    """
    
    @abstractmethod
    def get(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Retrieve a cached Pydantic model instance.
        
        Args:
            key: The cache key to look up
            model_class: The Pydantic model class to deserialize to
            
        Returns:
            The cached model instance if found, None otherwise
            
        Raises:
            Should handle all errors gracefully and return None on failure
        """
        pass
    
    @abstractmethod  
    def set(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Cache a Pydantic model instance.
        
        Args:
            key: The cache key to store under
            value: The Pydantic model instance to cache
            ttl: Optional time-to-live in seconds
            
        Raises:
            Should handle all errors gracefully and not raise exceptions
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a cached item.
        
        Args:
            key: The cache key to delete
            
        Raises:
            Should handle all errors gracefully and not raise exceptions
        """
        pass
    
    @abstractmethod
    async def aget(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Async version of get().
        
        Args:
            key: The cache key to look up
            model_class: The Pydantic model class to deserialize to
            
        Returns:
            The cached model instance if found, None otherwise
            
        Raises:
            Should handle all errors gracefully and return None on failure
        """
        pass
    
    @abstractmethod
    async def aset(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Async version of set().
        
        Args:
            key: The cache key to store under
            value: The Pydantic model instance to cache
            ttl: Optional time-to-live in seconds
            
        Raises:
            Should handle all errors gracefully and not raise exceptions
        """
        pass
    
    @abstractmethod
    async def adelete(self, key: str) -> None:
        """Async version of delete().
        
        Args:
            key: The cache key to delete
            
        Raises:
            Should handle all errors gracefully and not raise exceptions
        """
        pass
    
    def clear(self) -> None:
        """Clear all cached items.
        
        Optional method - implementations may override if supported.
        Default implementation does nothing.
        """
        pass
    
    async def aclear(self) -> None:
        """Async version of clear().
        
        Optional method - implementations may override if supported.
        Default implementation does nothing.
        """
        pass
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Optional method - implementations may override to provide metrics.
        
        Returns:
            Dictionary containing cache statistics like hit rate, size, etc.
            Default implementation returns empty dict.
        """
        return {}


class CacheError(Exception):
    """Base exception for cache-related errors.
    
    Cache implementations should catch and handle their specific errors
    gracefully, but may raise CacheError for unrecoverable issues.
    """
    pass


class CacheConnectionError(CacheError):
    """Raised when cache backend cannot connect to storage."""
    pass


class CacheSerializationError(CacheError):
    """Raised when Pydantic model cannot be serialized/deserialized."""
    pass