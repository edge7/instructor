"""
Caching module for Instructor.

This module provides various caching backends for the Instructor library,
enabling automatic caching of LLM responses with schema-aware invalidation.

Available cache backends:
- LRUCache: In-memory LRU cache
- RedisCache: Redis-based distributed cache  
- DiskCache: Persistent disk-based cache

Example usage:
    from instructor.cache import LRUCache, RedisCache, DiskCache
    import instructor
    
    # LRU Cache
    cache = LRUCache(maxsize=1000)
    client = instructor.from_provider("openai/gpt-4", cache=cache)
    
    # Redis Cache
    cache = RedisCache(redis_url="redis://localhost", ttl=3600)
    client = instructor.from_provider("openai/gpt-4", cache=cache)
    
    # Disk Cache
    cache = DiskCache(cache_dir="./cache", ttl=3600)
    client = instructor.from_provider("openai/gpt-4", cache=cache)
"""

from .base import CacheBackend
from .implementations import LRUCache, RedisCache, DiskCache

__all__ = [
    "CacheBackend",
    "LRUCache", 
    "RedisCache",
    "DiskCache",
]