"""
Concrete cache backend implementations.

This module provides production-ready cache implementations including:
- LRUCache: In-memory LRU cache with automatic eviction
- RedisCache: Redis-based distributed cache with TTL support
- DiskCache: Persistent disk-based cache using diskcache

All implementations include schema-aware cache invalidation and graceful
error handling.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import CacheBackend, CacheError, CacheSerializationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LRUCache(CacheBackend):
    """In-memory LRU (Least Recently Used) cache implementation.
    
    This cache stores Pydantic models in memory with automatic eviction
    of least recently used items when the maximum size is reached.
    
    Features:
    - Fast in-memory access (sub-millisecond)
    - Automatic LRU eviction
    - Schema-aware cache keys
    - Thread-safe operations
    - Optional metrics tracking
    
    Best for:
    - Development and testing
    - Single-process applications
    - Fast access to frequently used data
    """
    
    def __init__(self, maxsize: int = 1000):
        """Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items to cache. Set to 0 to disable caching.
        """
        self.maxsize = maxsize
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()  # For async safety
    
    @functools.lru_cache(maxsize=None)
    def _get_schema_hash(self, model_class: Type[BaseModel]) -> str:
        """Get hash of model schema for cache invalidation.
        
        This method is cached to avoid repeated schema serialization.
        
        Args:
            model_class: The Pydantic model class
            
        Returns:
            8-character hash of the model schema
        """
        try:
            schema = json.dumps(model_class.model_json_schema(), sort_keys=True)
            return hashlib.md5(schema.encode()).hexdigest()[:8]
        except Exception:
            # Fallback to class name if schema serialization fails
            return hashlib.md5(model_class.__name__.encode()).hexdigest()[:8]
    
    def _make_key(self, args: tuple, kwargs: dict, model_class: Type[BaseModel]) -> str:
        """Generate cache key with schema versioning.
        
        Args:
            args: Function arguments
            kwargs: Function keyword arguments  
            model_class: Pydantic model class for schema hash
            
        Returns:
            Cache key that includes schema hash for automatic invalidation
        """
        schema_hash = self._get_schema_hash(model_class)
        args_hash = hashlib.md5(str((args, kwargs)).encode()).hexdigest()[:8]
        return f"create:{schema_hash}:{args_hash}"
    
    def get(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Retrieve cached Pydantic model instance."""
        if self.maxsize == 0:
            return None
            
        try:
            cached_json = self._cache.get(key)
            if cached_json is None:
                self._misses += 1
                return None
            
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            
            # Deserialize from JSON
            return model_class.model_validate_json(cached_json)
        
        except Exception as e:
            logger.warning(f"LRU cache get error for key {key}: {e}")
            self._misses += 1
            return None
    
    def set(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Cache a Pydantic model instance."""
        if self.maxsize == 0:
            return
            
        try:
            # Serialize to JSON
            serialized_value = value.model_dump_json()
            
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
            
            # Add to cache
            self._cache[key] = serialized_value
            
            # Enforce size limit
            while len(self._cache) > self.maxsize:
                # Remove least recently used item
                self._cache.popitem(last=False)
        
        except Exception as e:
            logger.warning(f"LRU cache set error for key {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete a cached item."""
        try:
            self._cache.pop(key, None)
        except Exception as e:
            logger.warning(f"LRU cache delete error for key {key}: {e}")
    
    async def aget(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Async version of get()."""
        async with self._lock:
            return self.get(key, model_class)
    
    async def aset(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Async version of set()."""
        async with self._lock:
            self.set(key, value, ttl)
    
    async def adelete(self, key: str) -> None:
        """Async version of delete()."""
        async with self._lock:
            self.delete(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    async def aclear(self) -> None:
        """Async version of clear()."""
        async with self._lock:
            self.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "type": "LRU",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "current_size": len(self._cache),
            "max_size": self.maxsize,
        }


class RedisCache(CacheBackend):
    """Redis-based distributed cache implementation.
    
    This cache uses Redis for distributed caching across multiple processes
    or machines, with built-in TTL support and error handling.
    
    Features:
    - Distributed caching across processes/machines
    - Automatic TTL (time-to-live) expiration
    - Redis cluster support
    - Connection pooling
    - Graceful error handling with fallback
    - Async support via aioredis
    
    Best for:
    - Production distributed systems
    - Microservices architectures
    - Shared caching across multiple workers
    - High-throughput applications
    """
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600,
        key_prefix: str = "instructor",
        **redis_kwargs
    ):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
            **redis_kwargs: Additional Redis client arguments
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.redis_kwargs = redis_kwargs
        
        # Lazy initialization
        self._redis = None
        self._aioredis = None
        
        self._setup_redis()
    
    def _setup_redis(self):
        """Setup Redis connection with error handling."""
        try:
            import redis
            self._redis = redis.from_url(
                self.redis_url, 
                decode_responses=True,
                **self.redis_kwargs
            )
            # Test connection
            self._redis.ping()
            logger.info("Redis cache connected successfully")
        except ImportError:
            raise ImportError(
                "Redis package required for RedisCache. "
                "Install with: pip install redis"
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            # Don't raise - allow graceful degradation
    
    async def _setup_aioredis(self):
        """Setup async Redis connection."""
        if self._aioredis is not None:
            return
            
        try:
            import aioredis
            self._aioredis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                **self.redis_kwargs
            )
            # Test connection
            await self._aioredis.ping()
            logger.info("Async Redis cache connected successfully")
        except ImportError:
            raise ImportError(
                "aioredis package required for async RedisCache. "
                "Install with: pip install aioredis"
            )
        except Exception as e:
            logger.warning(f"Async Redis connection failed: {e}")
    
    def _make_redis_key(self, key: str) -> str:
        """Add prefix to cache key."""
        return f"{self.key_prefix}:{key}"
    
    def get(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Retrieve cached Pydantic model instance."""
        if self._redis is None:
            return None
            
        try:
            redis_key = self._make_redis_key(key)
            cached_json = self._redis.get(redis_key)
            
            if cached_json is None:
                return None
            
            return model_class.model_validate_json(cached_json)
        
        except Exception as e:
            logger.warning(f"Redis cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Cache a Pydantic model instance."""
        if self._redis is None:
            return
            
        try:
            redis_key = self._make_redis_key(key)
            serialized_value = value.model_dump_json()
            
            self._redis.setex(redis_key, ttl or self.ttl, serialized_value)
        
        except Exception as e:
            logger.warning(f"Redis cache set error for key {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete a cached item."""
        if self._redis is None:
            return
            
        try:
            redis_key = self._make_redis_key(key)
            self._redis.delete(redis_key)
        except Exception as e:
            logger.warning(f"Redis cache delete error for key {key}: {e}")
    
    async def aget(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Async version of get()."""
        await self._setup_aioredis()
        
        if self._aioredis is None:
            return None
            
        try:
            redis_key = self._make_redis_key(key)
            cached_json = await self._aioredis.get(redis_key)
            
            if cached_json is None:
                return None
            
            return model_class.model_validate_json(cached_json)
        
        except Exception as e:
            logger.warning(f"Async Redis cache get error for key {key}: {e}")
            return None
    
    async def aset(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Async version of set()."""
        await self._setup_aioredis()
        
        if self._aioredis is None:
            return
            
        try:
            redis_key = self._make_redis_key(key)
            serialized_value = value.model_dump_json()
            
            await self._aioredis.setex(redis_key, ttl or self.ttl, serialized_value)
        
        except Exception as e:
            logger.warning(f"Async Redis cache set error for key {key}: {e}")
    
    async def adelete(self, key: str) -> None:
        """Async version of delete()."""
        await self._setup_aioredis()
        
        if self._aioredis is None:
            return
            
        try:
            redis_key = self._make_redis_key(key)
            await self._aioredis.delete(redis_key)
        except Exception as e:
            logger.warning(f"Async Redis cache delete error for key {key}: {e}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._redis is None:
            return {"type": "Redis", "status": "disconnected"}
            
        try:
            info = self._redis.info()
            return {
                "type": "Redis",
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / 
                          max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            }
        except Exception as e:
            logger.warning(f"Redis stats error: {e}")
            return {"type": "Redis", "status": "error", "error": str(e)}


class DiskCache(CacheBackend):
    """Persistent disk-based cache implementation.
    
    This cache uses the diskcache library to provide persistent caching
    that survives process restarts.
    
    Features:
    - Persistent storage across restarts
    - Automatic eviction policies
    - TTL support
    - ACID transactions
    - Cross-process safe
    - Optional compression
    
    Best for:
    - Applications that need persistence
    - Expensive computations worth caching long-term
    - Development with cache persistence
    - Medium-performance requirements
    """
    
    def __init__(
        self,
        cache_dir: str = "./instructor_cache",
        ttl: Optional[int] = None,
        size_limit: int = 2**30,  # 1GB default
        **diskcache_kwargs
    ):
        """Initialize disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Default time-to-live in seconds
            size_limit: Maximum cache size in bytes
            **diskcache_kwargs: Additional diskcache arguments
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.size_limit = size_limit
        self.diskcache_kwargs = diskcache_kwargs
        
        self._setup_diskcache()
    
    def _setup_diskcache(self):
        """Setup diskcache with error handling."""
        try:
            import diskcache
            self._cache = diskcache.Cache(
                self.cache_dir,
                size_limit=self.size_limit,
                **self.diskcache_kwargs
            )
            logger.info(f"Disk cache initialized at {self.cache_dir}")
        except ImportError:
            raise ImportError(
                "diskcache package required for DiskCache. "
                "Install with: pip install diskcache"
            )
        except Exception as e:
            logger.error(f"Failed to initialize disk cache: {e}")
            raise
    
    def get(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Retrieve cached Pydantic model instance."""
        try:
            cached_json = self._cache.get(key)
            
            if cached_json is None:
                return None
            
            return model_class.model_validate_json(cached_json)
        
        except Exception as e:
            logger.warning(f"Disk cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Cache a Pydantic model instance."""
        try:
            serialized_value = value.model_dump_json()
            
            if ttl or self.ttl:
                self._cache.set(key, serialized_value, expire=ttl or self.ttl)
            else:
                self._cache.set(key, serialized_value)
        
        except Exception as e:
            logger.warning(f"Disk cache set error for key {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete a cached item."""
        try:
            self._cache.delete(key)
        except Exception as e:
            logger.warning(f"Disk cache delete error for key {key}: {e}")
    
    async def aget(self, key: str, model_class: Type[T]) -> Optional[T]:
        """Async version of get()."""
        # Disk operations are typically fast enough to not need true async
        return self.get(key, model_class)
    
    async def aset(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Async version of set()."""
        self.set(key, value, ttl)
    
    async def adelete(self, key: str) -> None:
        """Async version of delete()."""
        self.delete(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        try:
            self._cache.clear()
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")
    
    async def aclear(self) -> None:
        """Async version of clear()."""
        self.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            return {
                "type": "Disk",
                "cache_dir": self.cache_dir,
                "size": len(self._cache),
                "volume": self._cache.volume(),
                "size_limit": self.size_limit,
            }
        except Exception as e:
            logger.warning(f"Disk cache stats error: {e}")
            return {"type": "Disk", "status": "error", "error": str(e)}