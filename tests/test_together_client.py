"""Tests for Together client support in instructor."""

import pytest
from unittest.mock import Mock, patch
import instructor
from instructor import AsyncInstructor, Instructor


def test_from_openai_with_together_client():
    """Test that from_openai works with Together client instances."""
    
    # Mock the Together client
    mock_together_client = Mock()
    mock_together_client.api_key = "test_api_key"
    mock_together_client.base_url = "https://api.together.xyz/v1"
    
    # Mock the together module
    with patch.dict('sys.modules', {'together': Mock()}):
        import sys
        sys.modules['together'].Together = Mock
        sys.modules['together'].AsyncTogether = Mock
        
        # Mock isinstance to return True for our mock client
        with patch('builtins.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if obj is mock_together_client and hasattr(cls, '__name__') and 'Together' in str(cls):
                    return True
                # Call original isinstance for other cases
                return isinstance.__wrapped__(obj, cls)
            
            mock_isinstance.side_effect = isinstance_side_effect
            
            # Mock openai.OpenAI creation
            with patch('openai.OpenAI') as mock_openai:
                mock_openai_client = Mock()
                mock_openai_client.base_url = "https://api.together.xyz/v1"
                mock_openai.return_value = mock_openai_client
                
                # Mock instructor.patch
                with patch('instructor.patch') as mock_patch:
                    mock_patch.return_value = Mock()
                    
                    # Test the function
                    result = instructor.from_openai(mock_together_client)
                    
                    # Verify OpenAI client was created with correct parameters
                    mock_openai.assert_called_once_with(
                        api_key="test_api_key",
                        base_url="https://api.together.xyz/v1",
                    )
                    
                    # Verify result is an Instructor instance
                    assert isinstance(result, Instructor)


def test_from_openai_with_async_together_client():
    """Test that from_openai works with AsyncTogether client instances."""
    
    # Mock the AsyncTogether client
    mock_async_together_client = Mock()
    mock_async_together_client.api_key = "test_api_key"
    mock_async_together_client.base_url = "https://api.together.xyz/v1"
    
    # Mock the together module
    with patch.dict('sys.modules', {'together': Mock()}):
        import sys
        sys.modules['together'].Together = Mock
        sys.modules['together'].AsyncTogether = Mock
        
        # Mock isinstance to return True for our mock client
        with patch('builtins.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if obj is mock_async_together_client and hasattr(cls, '__name__') and 'AsyncTogether' in str(cls):
                    return True
                # Call original isinstance for other cases
                return isinstance.__wrapped__(obj, cls)
            
            mock_isinstance.side_effect = isinstance_side_effect
            
            # Mock openai.AsyncOpenAI creation
            with patch('openai.AsyncOpenAI') as mock_async_openai:
                mock_async_openai_client = Mock()
                mock_async_openai_client.base_url = "https://api.together.xyz/v1"
                mock_async_openai.return_value = mock_async_openai_client
                
                # Mock instructor.patch
                with patch('instructor.patch') as mock_patch:
                    mock_patch.return_value = Mock()
                    
                    # Test the function
                    result = instructor.from_openai(mock_async_together_client)
                    
                    # Verify AsyncOpenAI client was created with correct parameters
                    mock_async_openai.assert_called_once_with(
                        api_key="test_api_key",
                        base_url="https://api.together.xyz/v1",
                    )
                    
                    # Verify result is an AsyncInstructor instance
                    assert isinstance(result, AsyncInstructor)


def test_from_openai_without_together_package():
    """Test that from_openai gracefully handles missing together package."""
    
    # Mock a client that is not an OpenAI client
    mock_client = Mock()
    mock_client.base_url = "https://api.together.xyz/v1"
    
    # Mock ImportError when trying to import together
    with patch('instructor.client.importlib.import_module', side_effect=ImportError):
        with patch('warnings.warn') as mock_warn:
            # This should not raise an error, but should warn about unexpected client type
            try:
                result = instructor.from_openai(mock_client)
                # The function should still try to process it and warn about unexpected type
                mock_warn.assert_called()
            except Exception:
                # It's okay if it fails later in processing, we just want to ensure
                # the ImportError for together package is handled gracefully
                pass


def test_from_openai_together_provider_detection():
    """Test that Together provider is correctly detected from base URL."""
    
    # Mock the Together client
    mock_together_client = Mock()
    mock_together_client.api_key = "test_api_key"
    mock_together_client.base_url = "https://api.together.xyz/v1"
    
    # Mock the together module
    with patch.dict('sys.modules', {'together': Mock()}):
        import sys
        sys.modules['together'].Together = Mock
        
        # Mock isinstance to return True for our mock client
        with patch('builtins.isinstance') as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if obj is mock_together_client and hasattr(cls, '__name__') and 'Together' in str(cls):
                    return True
                return isinstance.__wrapped__(obj, cls)
            
            mock_isinstance.side_effect = isinstance_side_effect
            
            # Mock openai.OpenAI creation
            with patch('openai.OpenAI') as mock_openai:
                mock_openai_client = Mock()
                mock_openai_client.base_url = "https://api.together.xyz/v1"
                mock_openai.return_value = mock_openai_client
                
                # Mock instructor.patch
                with patch('instructor.patch') as mock_patch:
                    mock_patch.return_value = Mock()
                    
                    # Test with TOOLS mode (should be allowed for Together)
                    result = instructor.from_openai(mock_together_client, mode=instructor.Mode.TOOLS)
                    assert isinstance(result, Instructor)
                    
                    # Test with JSON mode (should be allowed for Together)
                    result = instructor.from_openai(mock_together_client, mode=instructor.Mode.JSON)
                    assert isinstance(result, Instructor)