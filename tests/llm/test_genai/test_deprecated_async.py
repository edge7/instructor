import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from instructor.client_genai import from_genai
from instructor.exceptions import ConfigurationError

class User(BaseModel):
    name: str
    age: int

@patch('instructor.client_genai.isinstance', return_value=True)
def test_deprecated_use_async_warning(_):
    """Test that using use_async parameter raises a deprecation warning."""
    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    
    with pytest.warns(DeprecationWarning, match="'use_async' is deprecated. Use 'async_client' instead."):
        client = from_genai(
            mock_client, 
            use_async=True
        )

@patch('instructor.client_genai.isinstance', return_value=True)
def test_both_async_params_error(_):
    """Test that providing both use_async and async_client raises an error."""
    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    
    with pytest.raises(ConfigurationError, match="Cannot provide both 'use_async' and 'async_client'. Use 'async_client' instead."):
        client = from_genai(
            mock_client, 
            use_async=True,
            async_client=True
        )