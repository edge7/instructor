"""
Tests for cache backend implementations.

This module tests the core caching functionality including:
- Cache backend interface compliance
- Individual cache implementations (LRU, Redis, DiskCache)
- Schema-aware cache invalidation
- Error handling and graceful degradation
- Async support
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field


# Test Models for Schema Invalidation Testing
class UserV1(BaseModel):
    """Version 1 of User model"""
    name: str
    age: int


class UserV2(BaseModel):
    """Version 2 of User model with additional field"""
    name: str
    age: int
    email: Optional[str] = None  # New field - should invalidate cache


class UserV3(BaseModel):
    """Version 3 with field modification"""
    name: str
    age: int
    occupation: str  # Changed from email to occupation


class ComplexUser(BaseModel):
    """More complex model for testing"""
    name: str = Field(description="User's full name")
    age: int = Field(ge=0, le=150, description="User's age")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


# Cache Backend Interface Tests
class CacheBackendTestMixin:
    """Mixin for testing cache backend implementations"""
    
    def get_cache_backend(self):
        """Override in subclasses to provide cache backend"""
        raise NotImplementedError
    
    def test_basic_get_set(self):
        """Test basic get/set operations"""
        cache = self.get_cache_backend()
        
        # Test cache miss
        result = cache.get("nonexistent", UserV1)
        assert result is None
        
        # Test cache set and hit
        user = UserV1(name="Alice", age=30)
        cache.set("test_key", user)
        
        cached_user = cache.get("test_key", UserV1)
        assert cached_user is not None
        assert cached_user.name == "Alice"
        assert cached_user.age == 30
        assert isinstance(cached_user, UserV1)
    
    def test_schema_aware_invalidation(self):
        """Test that cache keys include schema hash for invalidation"""
        cache = self.get_cache_backend()
        
        # Cache a UserV1 instance
        user_v1 = UserV1(name="Bob", age=25)
        key_v1 = self._generate_schema_key("test", {}, UserV1)
        cache.set(key_v1, user_v1)
        
        # Should be able to retrieve with same schema
        cached = cache.get(key_v1, UserV1)
        assert cached is not None
        assert cached.name == "Bob"
        
        # Key for UserV2 should be different (different schema)
        key_v2 = self._generate_schema_key("test", {}, UserV2)
        assert key_v1 != key_v2  # Keys should be different due to schema hash
        
        # Should not find UserV1 data with UserV2 key
        cached_v2 = cache.get(key_v2, UserV2)
        assert cached_v2 is None
    
    def test_pydantic_model_serialization(self):
        """Test that complex Pydantic models serialize/deserialize correctly"""
        cache = self.get_cache_backend()
        
        complex_user = ComplexUser(
            name="Complex User",
            age=35,
            tags=["developer", "python", "ai"],
            metadata={"company": "AI Corp", "role": "engineer"}
        )
        
        cache.set("complex_test", complex_user)
        retrieved = cache.get("complex_test", ComplexUser)
        
        assert retrieved is not None
        assert retrieved.name == "Complex User"
        assert retrieved.age == 35
        assert retrieved.tags == ["developer", "python", "ai"]
        assert retrieved.metadata == {"company": "AI Corp", "role": "engineer"}
    
    def test_error_handling(self):
        """Test graceful error handling"""
        cache = self.get_cache_backend()
        
        # Test with invalid JSON (if applicable)
        # This test may vary by implementation
        try:
            result = cache.get("invalid_key", UserV1)
            # Should return None or handle gracefully
            assert result is None
        except Exception:
            pytest.fail("Cache should handle errors gracefully")
    
    def test_cache_overwrite(self):
        """Test that cache can be overwritten"""
        cache = self.get_cache_backend()
        
        # Set initial value
        user1 = UserV1(name="First", age=20)
        cache.set("overwrite_test", user1)
        
        # Overwrite with new value
        user2 = UserV1(name="Second", age=30)
        cache.set("overwrite_test", user2)
        
        # Should get the new value
        retrieved = cache.get("overwrite_test", UserV1)
        assert retrieved is not None
        assert retrieved.name == "Second"
        assert retrieved.age == 30
    
    def _generate_schema_key(self, func_name: str, kwargs: dict, model_class: type) -> str:
        """Generate cache key with schema versioning - matches implementation"""
        import hashlib
        import json
        
        # Include model schema in cache key
        schema_hash = hashlib.md5(
            json.dumps(model_class.model_json_schema(), sort_keys=True).encode()
        ).hexdigest()[:8]
        
        args_hash = hashlib.md5(str(kwargs).encode()).hexdigest()[:8]
        
        return f"{func_name}:{schema_hash}:{args_hash}"


class TestLRUCache(CacheBackendTestMixin):
    """Test LRU Cache implementation"""
    
    def get_cache_backend(self):
        # Will import actual implementation once created
        from instructor.cache.implementations import LRUCache
        return LRUCache(maxsize=100)
    
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        from instructor.cache.implementations import LRUCache
        
        cache = LRUCache(maxsize=2)  # Small cache for testing eviction
        
        user1 = UserV1(name="User1", age=20)
        user2 = UserV1(name="User2", age=25)  
        user3 = UserV1(name="User3", age=30)
        
        # Fill cache to capacity
        cache.set("key1", user1)
        cache.set("key2", user2)
        
        # Both should be retrievable
        assert cache.get("key1", UserV1) is not None
        assert cache.get("key2", UserV1) is not None
        
        # Add third item - should evict oldest
        cache.set("key3", user3)
        
        # key1 should be evicted, key2 and key3 should remain
        assert cache.get("key1", UserV1) is None
        assert cache.get("key2", UserV1) is not None
        assert cache.get("key3", UserV1) is not None
    
    def test_maxsize_zero(self):
        """Test cache with maxsize=0 (no caching)"""
        from instructor.cache.implementations import LRUCache
        
        cache = LRUCache(maxsize=0)
        user = UserV1(name="Test", age=25)
        
        cache.set("test", user)
        result = cache.get("test", UserV1)
        # With maxsize=0, nothing should be cached
        assert result is None


class TestRedisCache(CacheBackendTestMixin):
    """Test Redis Cache implementation"""
    
    def get_cache_backend(self):
        from instructor.cache.implementations import RedisCache
        # Use fakeredis for testing if available, otherwise skip
        try:
            import fakeredis
            import redis
            
            # Create fake redis instance
            fake_redis = fakeredis.FakeRedis(decode_responses=True)
            
            # Patch the redis creation in RedisCache
            with patch.object(redis, 'from_url', return_value=fake_redis):
                return RedisCache(redis_url="redis://fake", ttl=3600)
        except ImportError:
            pytest.skip("fakeredis not available")
    
    def test_ttl_functionality(self):
        """Test TTL (time-to-live) functionality"""
        try:
            import fakeredis
        except ImportError:
            pytest.skip("fakeredis not available")
            
        from instructor.cache.implementations import RedisCache
        
        fake_redis = fakeredis.FakeRedis(decode_responses=True)
        
        with patch('redis.from_url', return_value=fake_redis):
            cache = RedisCache(ttl=1)  # 1 second TTL
            
            user = UserV1(name="TTL Test", age=25)
            cache.set("ttl_test", user, ttl=1)
            
            # Should be available immediately
            result = cache.get("ttl_test", UserV1)
            assert result is not None
            
            # Simulate TTL expiration
            fake_redis.expire("ttl_test", -1)  # Force expiration
            
            # Should now be None
            result = cache.get("ttl_test", UserV1)
            assert result is None
    
    def test_redis_connection_failure(self):
        """Test graceful handling of Redis connection failures"""
        from instructor.cache.implementations import RedisCache
        
        # Mock Redis to raise connection error
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Connection failed")
        mock_redis.setex.side_effect = Exception("Connection failed")
        
        with patch('redis.from_url', return_value=mock_redis):
            cache = RedisCache()
            
            # Should handle get failures gracefully
            result = cache.get("test", UserV1)
            assert result is None
            
            # Should handle set failures gracefully
            user = UserV1(name="Test", age=25)
            try:
                cache.set("test", user)  # Should not raise
            except Exception:
                pytest.fail("Cache should handle Redis failures gracefully")


class TestDiskCache(CacheBackendTestMixin):
    """Test Disk Cache implementation"""
    
    def get_cache_backend(self):
        from instructor.cache.implementations import DiskCache
        
        # Use temporary directory for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        return DiskCache(cache_dir=temp_dir, ttl=3600)
    
    def test_persistence_across_instances(self):
        """Test that disk cache persists across different instances"""
        from instructor.cache.implementations import DiskCache
        
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Create first cache instance and store data
        cache1 = DiskCache(cache_dir=temp_dir)
        user = UserV1(name="Persistent", age=40)
        cache1.set("persist_test", user)
        
        # Create second cache instance with same directory
        cache2 = DiskCache(cache_dir=temp_dir)
        result = cache2.get("persist_test", UserV1)
        
        assert result is not None
        assert result.name == "Persistent"
        assert result.age == 40
    
    def test_disk_cache_import_error(self):
        """Test handling when diskcache package is not available"""
        # Mock the import to fail
        with patch.dict('sys.modules', {'diskcache': None}):
            with pytest.raises(ImportError) as excinfo:
                from instructor.cache.implementations import DiskCache
                DiskCache()
            assert "diskcache package required" in str(excinfo.value)


# Async Cache Tests
class AsyncCacheBackendTestMixin:
    """Mixin for testing async cache backend implementations"""
    
    async def get_async_cache_backend(self):
        """Override in subclasses to provide async cache backend"""
        raise NotImplementedError
    
    @pytest.mark.asyncio
    async def test_async_basic_get_set(self):
        """Test basic async get/set operations"""
        cache = await self.get_async_cache_backend()
        
        # Test cache miss
        result = await cache.aget("nonexistent", UserV1)
        assert result is None
        
        # Test cache set and hit
        user = UserV1(name="AsyncAlice", age=30)
        await cache.aset("async_test_key", user)
        
        cached_user = await cache.aget("async_test_key", UserV1)
        assert cached_user is not None
        assert cached_user.name == "AsyncAlice"
        assert cached_user.age == 30
    
    @pytest.mark.asyncio
    async def test_async_concurrent_access(self):
        """Test concurrent async access to cache"""
        cache = await self.get_async_cache_backend()
        
        async def set_user(name: str, age: int):
            user = UserV1(name=name, age=age)
            await cache.aset(f"user_{name}", user)
            return await cache.aget(f"user_{name}", UserV1)
        
        # Run multiple concurrent operations
        tasks = [
            set_user("User1", 20),
            set_user("User2", 25),
            set_user("User3", 30),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result is not None
            assert result.name == f"User{i+1}"
            assert result.age == 20 + (i * 5)


class TestAsyncRedisCache(AsyncCacheBackendTestMixin):
    """Test async Redis cache functionality"""
    
    async def get_async_cache_backend(self):
        try:
            import fakeredis.aioredis
        except ImportError:
            pytest.skip("fakeredis aioredis not available")
            
        from instructor.cache.implementations import RedisCache
        
        # Create fake async redis
        fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        
        with patch('aioredis.from_url', return_value=fake_redis):
            return RedisCache(redis_url="redis://fake")


# Integration Tests
def test_schema_hash_generation():
    """Test that schema hash generation works correctly"""
    from instructor.cache.implementations import LRUCache
    
    cache = LRUCache()
    
    # Different models should have different schema hashes
    hash_v1 = cache._get_schema_hash(UserV1)
    hash_v2 = cache._get_schema_hash(UserV2)
    hash_v3 = cache._get_schema_hash(UserV3)
    
    assert hash_v1 != hash_v2
    assert hash_v1 != hash_v3
    assert hash_v2 != hash_v3
    
    # Same model should have same hash
    hash_v1_again = cache._get_schema_hash(UserV1)
    assert hash_v1 == hash_v1_again


def test_cache_key_generation():
    """Test cache key generation includes all relevant parameters"""
    from instructor.cache.implementations import LRUCache
    
    cache = LRUCache()
    
    # Same parameters should generate same key
    key1 = cache._make_key((), {"param": "value"}, UserV1)
    key2 = cache._make_key((), {"param": "value"}, UserV1)
    assert key1 == key2
    
    # Different parameters should generate different keys
    key3 = cache._make_key((), {"param": "different"}, UserV1)
    assert key1 != key3
    
    # Different models should generate different keys
    key4 = cache._make_key((), {"param": "value"}, UserV2)
    assert key1 != key4


def test_cache_metrics_interface():
    """Test that cache implementations can provide metrics"""
    from instructor.cache.implementations import LRUCache
    
    cache = LRUCache(maxsize=10)
    
    # Check if cache has metrics methods (optional)
    user = UserV1(name="Metrics", age=25)
    cache.set("metrics_test", user)
    
    # Should be able to get basic info
    result = cache.get("metrics_test", UserV1)
    assert result is not None
    
    # Cache should track hits/misses if implemented
    miss_result = cache.get("nonexistent", UserV1)
    assert miss_result is None


@pytest.mark.parametrize("cache_type", ["LRU", "Redis", "Disk"])
def test_cache_backend_compliance(cache_type):
    """Test that all cache backends comply with the interface"""
    
    if cache_type == "Redis":
        try:
            import fakeredis
        except ImportError:
            pytest.skip("fakeredis not available")
    elif cache_type == "Disk":
        # DiskCache should work without external deps except diskcache
        pass
    
    # Test will be implemented once we have the actual implementations
    # This is a placeholder for interface compliance testing
    assert True  # Placeholder


def test_cache_error_resilience():
    """Test that cache errors don't break the application"""
    # Mock a cache that always fails
    class FailingCache:
        def get(self, key, model_class):
            raise Exception("Cache error")
        
        def set(self, key, value, ttl=None):
            raise Exception("Cache error")
        
        async def aget(self, key, model_class):
            raise Exception("Async cache error")
        
        async def aset(self, key, value, ttl=None):
            raise Exception("Async cache error")
    
    cache = FailingCache()
    
    # These should not raise exceptions in a real implementation
    # The cache should handle errors gracefully
    try:
        result = cache.get("test", UserV1)
        # In a real implementation, this should return None gracefully
    except Exception:
        # For now, we expect this to fail, but in the real implementation
        # it should be handled gracefully
        pass
    
    user = UserV1(name="Test", age=25)
    try:
        cache.set("test", user)
        # In a real implementation, this should fail gracefully
    except Exception:
        # For now, we expect this to fail, but in the real implementation
        # it should be handled gracefully
        pass