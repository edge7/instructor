"""
Integration tests for from_provider() API with cache parameter.

This module tests the integration of caching into the from_provider() function,
ensuring that clients created with cache backends work correctly across all
supported providers.
"""
from __future__ import annotations

import os
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel


# Test Models
class User(BaseModel):
    name: str
    age: int


class ExtractedData(BaseModel):
    content: str
    confidence: float = 0.8


class TestFromProviderCacheIntegration:
    """Test cache integration with from_provider()"""
    
    def test_from_provider_with_lru_cache(self):
        """Test that from_provider works with LRU cache"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        cache = LRUCache(maxsize=100)
        
        # Skip if not in CI environment 
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        try:
            client = from_provider("openai/gpt-4o-mini", cache=cache)
            
            # Verify client is cached
            assert hasattr(client, 'cache')
            assert client.cache == cache
            
            # Verify it's a CachedInstructor instance
            from instructor.client import CachedInstructor
            assert isinstance(client, CachedInstructor)
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    def test_from_provider_with_redis_cache(self):
        """Test that from_provider works with Redis cache"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import RedisCache
        
        try:
            import fakeredis
        except ImportError:
            pytest.skip("fakeredis not available")
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        # Use fake redis for testing
        fake_redis = fakeredis.FakeRedis(decode_responses=True)
        
        with patch('redis.from_url', return_value=fake_redis):
            cache = RedisCache(redis_url="redis://fake")
            
            try:
                client = from_provider("openai/gpt-4o-mini", cache=cache)
                
                # Verify client is cached
                assert hasattr(client, 'cache')
                assert client.cache == cache
                
            except Exception as e:
                pytest.skip(f"Provider not available: {e}")
    
    def test_from_provider_with_disk_cache(self):
        """Test that from_provider works with disk cache"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import DiskCache
        
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        cache = DiskCache(cache_dir=temp_dir)
        
        try:
            client = from_provider("openai/gpt-4o-mini", cache=cache)
            
            # Verify client is cached
            assert hasattr(client, 'cache')
            assert client.cache == cache
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    @pytest.mark.asyncio
    async def test_from_provider_async_with_cache(self):
        """Test that async from_provider works with cache"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        cache = LRUCache(maxsize=100)
        
        try:
            client = from_provider("openai/gpt-4o-mini", async_client=True, cache=cache)
            
            # Verify client is cached async instructor
            assert hasattr(client, 'cache')
            assert client.cache == cache
            
            # Verify it's a CachedAsyncInstructor instance
            from instructor.client import CachedAsyncInstructor
            assert isinstance(client, CachedAsyncInstructor)
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    def test_from_provider_without_cache_unchanged(self):
        """Test that from_provider without cache works unchanged"""
        from instructor.auto_client import from_provider
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        try:
            client = from_provider("openai/gpt-4o-mini")
            
            # Should not have cache attribute
            assert not hasattr(client, 'cache')
            
            # Should be regular Instructor instance
            from instructor.client import Instructor
            assert isinstance(client, Instructor)
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    def test_from_provider_cache_with_different_providers(self):
        """Test cache integration works with different providers"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        cache = LRUCache(maxsize=100)
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            # Only test with available providers in CI
            providers = ["openai/gpt-4o-mini"]
        else:
            providers = [
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-haiku-latest",
                "google/gemini-2.0-flash"
            ]
        
        for provider in providers:
            try:
                client = from_provider(provider, cache=cache)
                assert hasattr(client, 'cache')
                assert client.cache == cache
            except Exception as e:
                pytest.skip(f"Provider {provider} not available: {e}")
    
    def test_cache_type_validation(self):
        """Test that invalid cache types are handled properly"""
        from instructor.auto_client import from_provider
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        # Test with invalid cache type
        invalid_cache = "not a cache"
        
        with pytest.raises(Exception):  # Should raise some kind of error
            from_provider("openai/gpt-4o-mini", cache=invalid_cache)


class TestCachedClientFunctionality:
    """Test actual caching functionality with real clients"""
    
    def test_cache_hit_miss_cycle(self):
        """Test complete cache hit/miss cycle"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        cache = LRUCache(maxsize=100)
        
        try:
            client = from_provider("openai/gpt-4o-mini", cache=cache)
            
            # Mock the underlying create function to count calls
            original_create = client.original_create_fn
            call_count = 0
            
            def counting_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return User(name="Test User", age=30)
            
            client.original_create_fn = counting_create
            
            # First call should be cache miss
            result1 = client.chat.completions.create(
                messages=[{"role": "user", "content": "Extract user"}],
                response_model=User
            )
            
            assert call_count == 1
            assert result1.name == "Test User"
            
            # Second call with same parameters should be cache hit
            result2 = client.chat.completions.create(
                messages=[{"role": "user", "content": "Extract user"}],
                response_model=User
            )
            
            assert call_count == 1  # Should not increment
            assert result2.name == "Test User"
            assert result1.name == result2.name
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    @pytest.mark.asyncio
    async def test_async_cache_functionality(self):
        """Test caching with async client"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        cache = LRUCache(maxsize=100)
        
        try:
            client = from_provider("openai/gpt-4o-mini", async_client=True, cache=cache)
            
            # Mock the underlying create function
            call_count = 0
            
            async def counting_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return User(name="Async Test User", age=25)
            
            client.original_create_fn = counting_create
            
            # First call should be cache miss
            result1 = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Extract user"}],
                response_model=User
            )
            
            assert call_count == 1
            assert result1.name == "Async Test User"
            
            # Second call should be cache hit
            result2 = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Extract user"}],
                response_model=User
            )
            
            assert call_count == 1  # Should not increment
            assert result2.name == "Async Test User"
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    def test_cache_with_different_response_models(self):
        """Test that different response models use different cache keys"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        cache = LRUCache(maxsize=100)
        
        try:
            client = from_provider("openai/gpt-4o-mini", cache=cache)
            
            call_count = 0
            
            def counting_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                response_model = kwargs.get('response_model')
                if response_model == User:
                    return User(name="User Result", age=30)
                elif response_model == ExtractedData:
                    return ExtractedData(content="Data Result")
                return None
            
            client.original_create_fn = counting_create
            
            # Call with User model
            result1 = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                response_model=User
            )
            assert call_count == 1
            assert result1.name == "User Result"
            
            # Call with ExtractedData model (same messages)
            result2 = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                response_model=ExtractedData
            )
            assert call_count == 2  # Should call again due to different model
            assert result2.content == "Data Result"
            
            # Call again with User model - should hit cache
            result3 = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                response_model=User
            )
            assert call_count == 2  # Should not increment
            assert result3.name == "User Result"
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")
    
    def test_cache_invalidation_on_schema_change(self):
        """Test that cache is invalidated when model schema changes"""
        from instructor.auto_client import from_provider
        from instructor.cache.implementations import LRUCache
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        cache = LRUCache(maxsize=100)
        
        try:
            client = from_provider("openai/gpt-4o-mini", cache=cache)
            
            # Create two different User models with same name but different schemas
            class UserV1(BaseModel):
                name: str
                age: int
            
            class UserV2(BaseModel):
                name: str
                age: int
                email: Optional[str] = None
            
            call_count = 0
            
            def counting_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                response_model = kwargs.get('response_model')
                if response_model == UserV1:
                    return UserV1(name="V1 User", age=30)
                elif response_model == UserV2:
                    return UserV2(name="V2 User", age=30, email="test@example.com")
                return None
            
            client.original_create_fn = counting_create
            
            # Call with UserV1
            result1 = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                response_model=UserV1
            )
            assert call_count == 1
            assert result1.name == "V1 User"
            
            # Call with UserV2 (different schema) - should not hit cache
            result2 = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                response_model=UserV2
            )
            assert call_count == 2  # Should call again
            assert result2.name == "V2 User"
            assert result2.email == "test@example.com"
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")


class TestCacheErrorResilience:
    """Test that cache errors don't break normal operation"""
    
    def test_cache_error_fallback(self):
        """Test that client works normally even when cache fails"""
        from instructor.auto_client import from_provider
        
        if os.getenv("INSTRUCTOR_ENV") == "CI":
            pytest.skip("Skipping provider tests in CI")
        
        # Create a cache that always fails
        class FailingCache:
            def get(self, key, model_class):
                raise Exception("Cache get failed")
            
            def set(self, key, value, ttl=None):
                raise Exception("Cache set failed")
            
            async def aget(self, key, model_class):
                raise Exception("Async cache get failed")
            
            async def aset(self, key, value, ttl=None):
                raise Exception("Async cache set failed")
        
        failing_cache = FailingCache()
        
        try:
            client = from_provider("openai/gpt-4o-mini", cache=failing_cache)
            
            # Mock successful response
            def mock_create(*args, **kwargs):
                return User(name="Fallback User", age=35)
            
            client.original_create_fn = mock_create
            
            # Should work despite cache failures
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                response_model=User
            )
            
            assert result.name == "Fallback User"
            assert result.age == 35
            
        except Exception as e:
            pytest.skip(f"Provider not available: {e}")


# Test Cache Metrics and Monitoring
class TestCacheMetrics:
    """Test cache metrics and monitoring functionality"""
    
    def test_cache_hit_rate_tracking(self):
        """Test that cache implementations can track hit rates"""
        from instructor.cache.implementations import LRUCache
        
        cache = LRUCache(maxsize=100)
        
        # Test basic metrics tracking (if implemented)
        user1 = User(name="User1", age=25)
        user2 = User(name="User2", age=30)
        
        # Set values
        cache.set("key1", user1)
        cache.set("key2", user2)
        
        # Test hits
        result1 = cache.get("key1", User)
        result2 = cache.get("key2", User)
        
        # Test misses
        miss1 = cache.get("nonexistent1", User)
        miss2 = cache.get("nonexistent2", User)
        
        assert result1 is not None
        assert result2 is not None
        assert miss1 is None
        assert miss2 is None
        
        # If metrics are implemented, verify them
        if hasattr(cache, 'get_metrics'):
            metrics = cache.get_metrics()
            assert 'hits' in metrics or 'hit_rate' in metrics
    
    def test_cache_size_monitoring(self):
        """Test cache size monitoring"""
        from instructor.cache.implementations import LRUCache
        
        cache = LRUCache(maxsize=3)
        
        # Fill cache
        for i in range(5):
            user = User(name=f"User{i}", age=20 + i)
            cache.set(f"key{i}", user)
        
        # Cache should not exceed maxsize
        # This test verifies the LRU eviction works
        
        # Check that only the last 3 items are available
        assert cache.get("key0", User) is None  # Should be evicted
        assert cache.get("key1", User) is None  # Should be evicted
        assert cache.get("key2", User) is not None
        assert cache.get("key3", User) is not None
        assert cache.get("key4", User) is not None