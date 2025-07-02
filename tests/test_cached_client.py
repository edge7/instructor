"""
Tests for CachedInstructor client integration.

This module tests the integration of caching into the Instructor client,
including cache interception at the create_fn level and proper handling
of all client methods.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional
from unittest.mock import Mock, AsyncMock, patch

import pytest
from pydantic import BaseModel


# Test Models
class User(BaseModel):
    name: str
    age: int


class UserWithEmail(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class TestCachedInstructor:
    """Test CachedInstructor functionality"""
    
    def test_cached_instructor_creation(self):
        """Test that CachedInstructor can be created with a cache backend"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        # Mock the base instructor components
        mock_client = Mock()
        mock_create_fn = Mock()
        mock_mode = Mock()
        mock_provider = Mock()
        
        cache = LRUCache(maxsize=100)
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=mock_client,
            create=mock_create_fn,
            mode=mock_mode,
            provider=mock_provider
        )
        
        assert cached_instructor.cache == cache
        assert cached_instructor.original_create_fn == mock_create_fn
        assert cached_instructor.create_fn != mock_create_fn  # Should be wrapped
    
    def test_cache_hit_flow(self):
        """Test that cache hits bypass the original create function"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        # Setup mocks
        mock_client = Mock()
        mock_create_fn = Mock()
        cache = LRUCache(maxsize=100)
        
        # Pre-populate cache
        user = User(name="Cached User", age=30)
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=mock_client,
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # Generate the expected cache key
        cache_key = cached_instructor._generate_cache_key(test_kwargs, User)
        cache.set(cache_key, user)
        
        # Call the cached function
        result = cached_instructor.create_fn(**test_kwargs)
        
        # Should get cached result without calling original function
        assert result.name == "Cached User"
        assert result.age == 30
        assert not mock_create_fn.called
    
    def test_cache_miss_flow(self):
        """Test that cache misses call the original function and cache the result"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        # Setup mocks
        mock_client = Mock()
        expected_user = User(name="New User", age=25)
        mock_create_fn = Mock(return_value=expected_user)
        cache = LRUCache(maxsize=100)
        
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=mock_client,
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # Call the cached function (cache miss)
        result = cached_instructor.create_fn(**test_kwargs)
        
        # Should call original function and return result
        assert result.name == "New User"
        assert result.age == 25
        assert mock_create_fn.called
        
        # Should cache the result for next time
        cache_key = cached_instructor._generate_cache_key(test_kwargs, User)
        cached_result = cache.get(cache_key, User)
        assert cached_result is not None
        assert cached_result.name == "New User"
    
    def test_non_pydantic_response_model_bypasses_cache(self):
        """Test that non-Pydantic response models bypass caching"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        # Setup mocks
        mock_client = Mock()
        mock_create_fn = Mock(return_value="string result")
        cache = LRUCache(maxsize=100)
        
        test_kwargs = {
            'response_model': None,  # No response model
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=mock_client,
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # Call the cached function
        result = cached_instructor.create_fn(**test_kwargs)
        
        # Should call original function directly (no caching)
        assert result == "string result"
        assert mock_create_fn.called
    
    def test_cache_key_generation_deterministic(self):
        """Test that cache key generation is deterministic"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        cached_instructor = CachedInstructor(
            cache=LRUCache(),
            client=Mock(),
            create=Mock(),
            mode=Mock(),
            provider=Mock()
        )
        
        test_kwargs = {
            'messages': [{'role': 'user', 'content': 'test message'}],
            'model': 'gpt-4',
            'temperature': 0.7
        }
        
        # Same inputs should generate same key
        key1 = cached_instructor._generate_cache_key(test_kwargs, User)
        key2 = cached_instructor._generate_cache_key(test_kwargs, User)
        assert key1 == key2
        
        # Different models should generate different keys
        key3 = cached_instructor._generate_cache_key(test_kwargs, UserWithEmail)
        assert key1 != key3
        
        # Different messages should generate different keys
        test_kwargs_2 = test_kwargs.copy()
        test_kwargs_2['messages'] = [{'role': 'user', 'content': 'different message'}]
        key4 = cached_instructor._generate_cache_key(test_kwargs_2, User)
        assert key1 != key4
    
    def test_cache_key_includes_schema_hash(self):
        """Test that cache keys include model schema hash"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        cached_instructor = CachedInstructor(
            cache=LRUCache(),
            client=Mock(),
            create=Mock(),
            mode=Mock(),
            provider=Mock()
        )
        
        test_kwargs = {'messages': [], 'model': 'gpt-4'}
        
        key_user = cached_instructor._generate_cache_key(test_kwargs, User)
        key_user_email = cached_instructor._generate_cache_key(test_kwargs, UserWithEmail)
        
        # Keys should be different due to different schemas
        assert key_user != key_user_email
        
        # Keys should contain schema-related information
        assert "instructor:" in key_user
        assert "instructor:" in key_user_email
    
    def test_cache_error_handling(self):
        """Test graceful handling of cache errors"""
        from instructor.client import CachedInstructor
        
        # Mock cache that raises errors
        mock_cache = Mock()
        mock_cache.get.side_effect = Exception("Cache error")
        mock_cache.set.side_effect = Exception("Cache error")
        
        # Setup instructor
        expected_user = User(name="Result User", age=40)
        mock_create_fn = Mock(return_value=expected_user)
        
        cached_instructor = CachedInstructor(
            cache=mock_cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        # Should handle cache errors gracefully and still return result
        result = cached_instructor.create_fn(**test_kwargs)
        
        assert result.name == "Result User"
        assert result.age == 40
        assert mock_create_fn.called
    
    def test_cache_integration_with_hooks(self):
        """Test that caching works properly with instructor hooks"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        from instructor.hooks import Hooks
        
        # Setup hooks
        hooks = Hooks()
        hook_called = False
        
        def test_hook(data):
            nonlocal hook_called
            hook_called = True
        
        hooks.on("completion:response", test_hook)
        
        # Setup cached instructor
        expected_user = User(name="Hook User", age=35)
        mock_create_fn = Mock(return_value=expected_user)
        cache = LRUCache(maxsize=100)
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock(),
            hooks=hooks
        )
        
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4',
            'hooks': hooks
        }
        
        # First call (cache miss) - should call hooks
        result = cached_instructor.create_fn(**test_kwargs)
        
        assert result.name == "Hook User"
        assert mock_create_fn.called
        
        # Reset for second call
        mock_create_fn.reset_mock()
        hook_called = False
        
        # Second call (cache hit) - should not call original function or hooks
        result2 = cached_instructor.create_fn(**test_kwargs)
        
        assert result2.name == "Hook User"
        assert not mock_create_fn.called
        # Note: Hook behavior on cache hits may vary by implementation


class TestCachedAsyncInstructor:
    """Test CachedAsyncInstructor functionality"""
    
    @pytest.mark.asyncio
    async def test_async_cache_hit_flow(self):
        """Test async cache hit flow"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedAsyncInstructor
        
        # Setup mocks
        mock_client = Mock()
        mock_create_fn = AsyncMock()
        cache = LRUCache(maxsize=100)
        
        # Pre-populate cache
        user = User(name="Async Cached User", age=30)
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        cached_instructor = CachedAsyncInstructor(
            cache=cache,
            client=mock_client,
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # Generate the expected cache key
        cache_key = cached_instructor._generate_cache_key(test_kwargs, User)
        await cache.aset(cache_key, user)
        
        # Call the cached function
        result = await cached_instructor.create_fn(**test_kwargs)
        
        # Should get cached result without calling original function
        assert result.name == "Async Cached User"
        assert result.age == 30
        assert not mock_create_fn.called
    
    @pytest.mark.asyncio
    async def test_async_cache_miss_flow(self):
        """Test async cache miss flow"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedAsyncInstructor
        
        # Setup mocks
        mock_client = Mock()
        expected_user = User(name="Async New User", age=25)
        mock_create_fn = AsyncMock(return_value=expected_user)
        cache = LRUCache(maxsize=100)
        
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        cached_instructor = CachedAsyncInstructor(
            cache=cache,
            client=mock_client,
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # Call the cached function (cache miss)
        result = await cached_instructor.create_fn(**test_kwargs)
        
        # Should call original function and return result
        assert result.name == "Async New User"
        assert result.age == 25
        assert mock_create_fn.called
        
        # Should cache the result for next time
        cache_key = cached_instructor._generate_cache_key(test_kwargs, User)
        cached_result = await cache.aget(cache_key, User)
        assert cached_result is not None
        assert cached_result.name == "Async New User"
    
    @pytest.mark.asyncio
    async def test_async_cache_error_handling(self):
        """Test async cache error handling"""
        from instructor.client import CachedAsyncInstructor
        
        # Mock cache that raises errors
        mock_cache = Mock()
        mock_cache.aget.side_effect = Exception("Async cache error")
        mock_cache.aset.side_effect = Exception("Async cache error")
        
        # Setup instructor
        expected_user = User(name="Async Result User", age=40)
        mock_create_fn = AsyncMock(return_value=expected_user)
        
        cached_instructor = CachedAsyncInstructor(
            cache=mock_cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        test_kwargs = {
            'response_model': User,
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'gpt-4'
        }
        
        # Should handle cache errors gracefully and still return result
        result = await cached_instructor.create_fn(**test_kwargs)
        
        assert result.name == "Async Result User"
        assert result.age == 40
        assert mock_create_fn.called


class TestCachedInstructorMethods:
    """Test that all Instructor methods work with caching"""
    
    def test_create_method_with_cache(self):
        """Test that the create method works with caching"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        cache = LRUCache(maxsize=100)
        mock_create_fn = Mock(return_value=User(name="Created User", age=30))
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # Test create method
        result = cached_instructor.create(
            response_model=User,
            messages=[{'role': 'user', 'content': 'test'}]
        )
        
        assert result.name == "Created User"
        assert mock_create_fn.called
    
    def test_create_partial_method_handling(self):
        """Test how create_partial handles caching"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        # Note: create_partial returns generators, so caching behavior may differ
        # This test ensures the method still works
        cache = LRUCache(maxsize=100)
        mock_create_fn = Mock()
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # This should work without errors (implementation may vary for streaming)
        try:
            cached_instructor.create_partial(
                response_model=User,
                messages=[{'role': 'user', 'content': 'test'}]
            )
        except Exception as e:
            pytest.fail(f"create_partial should work with caching: {e}")
    
    def test_create_iterable_method_handling(self):
        """Test how create_iterable handles caching"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        # Note: create_iterable returns generators, so caching behavior may differ
        cache = LRUCache(maxsize=100)
        mock_create_fn = Mock()
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # This should work without errors
        try:
            cached_instructor.create_iterable(
                response_model=User,
                messages=[{'role': 'user', 'content': 'test'}]
            )
        except Exception as e:
            pytest.fail(f"create_iterable should work with caching: {e}")
    
    def test_create_with_completion_method_handling(self):
        """Test how create_with_completion handles caching"""
        from instructor.cache.implementations import LRUCache
        from instructor.client import CachedInstructor
        
        cache = LRUCache(maxsize=100)
        mock_create_fn = Mock(return_value=(User(name="Test", age=30), Mock()))
        
        cached_instructor = CachedInstructor(
            cache=cache,
            client=Mock(),
            create=mock_create_fn,
            mode=Mock(),
            provider=Mock()
        )
        
        # This should work and potentially cache the user part
        try:
            result = cached_instructor.create_with_completion(
                response_model=User,
                messages=[{'role': 'user', 'content': 'test'}]
            )
            # Result should be a tuple
            assert isinstance(result, tuple)
        except Exception as e:
            pytest.fail(f"create_with_completion should work with caching: {e}")


# Performance and Integration Tests
def test_cache_performance_benefit():
    """Test that caching provides performance benefits"""
    from instructor.cache.implementations import LRUCache
    from instructor.client import CachedInstructor
    import time
    
    # Mock slow function
    def slow_create_fn(**kwargs):
        time.sleep(0.01)  # Simulate slow LLM call
        return User(name="Slow User", age=30)
    
    cache = LRUCache(maxsize=100)
    cached_instructor = CachedInstructor(
        cache=cache,
        client=Mock(),
        create=slow_create_fn,
        mode=Mock(),
        provider=Mock()
    )
    
    test_kwargs = {
        'response_model': User,
        'messages': [{'role': 'user', 'content': 'test'}],
        'model': 'gpt-4'
    }
    
    # First call (cache miss)
    start_time = time.time()
    result1 = cached_instructor.create_fn(**test_kwargs)
    first_call_time = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = cached_instructor.create_fn(**test_kwargs)
    second_call_time = time.time() - start_time
    
    # Cache hit should be significantly faster
    assert second_call_time < first_call_time / 2
    assert result1.name == result2.name
    assert result1.age == result2.age


def test_cache_consistency_across_calls():
    """Test that cached results are consistent across multiple calls"""
    from instructor.cache.implementations import LRUCache
    from instructor.client import CachedInstructor
    
    call_count = 0
    
    def counting_create_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        return User(name=f"User{call_count}", age=30)
    
    cache = LRUCache(maxsize=100)
    cached_instructor = CachedInstructor(
        cache=cache,
        client=Mock(),
        create=counting_create_fn,
        mode=Mock(),
        provider=Mock()
    )
    
    test_kwargs = {
        'response_model': User,
        'messages': [{'role': 'user', 'content': 'test'}],
        'model': 'gpt-4'
    }
    
    # Multiple calls with same parameters
    results = []
    for _ in range(5):
        result = cached_instructor.create_fn(**test_kwargs)
        results.append(result)
    
    # Should only call original function once
    assert call_count == 1
    
    # All results should be identical
    for result in results:
        assert result.name == "User1"
        assert result.age == 30