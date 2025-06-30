import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from instructor.client_bedrock import from_bedrock
from instructor.exceptions import ConfigurationError

class User(BaseModel):
    name: str
    age: int

@patch('instructor.client_bedrock.isinstance', return_value=True)
def test_deprecated_async_warning(_):
    """Test that using _async parameter raises a deprecation warning."""
    mock_client = MagicMock()
    mock_client.converse = MagicMock()
    
    with pytest.warns(DeprecationWarning, match="'_async' is deprecated. Use 'async_client' instead."):
        client = from_bedrock(
            mock_client, 
            _async=True
        )

@patch('instructor.client_bedrock.isinstance', return_value=True)
def test_both_async_params_error(_):
    """Test that providing both _async and async_client raises an error."""
    mock_client = MagicMock()
    mock_client.converse = MagicMock()
    
    with pytest.raises(ConfigurationError, match="Cannot provide both '_async' and 'async_client'. Use 'async_client' instead."):
        client = from_bedrock(
            mock_client, 
            _async=True,
            async_client=True
        )