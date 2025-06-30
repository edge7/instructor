import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from instructor.client_vertexai import from_vertexai
from instructor.exceptions import ConfigurationError

class User(BaseModel):
    name: str
    age: int

@patch('instructor.client_vertexai.isinstance', return_value=True)
def test_deprecated_async_warning(_):
    """Test that using _async parameter raises a deprecation warning."""
    mock_model = MagicMock()
    mock_model.generate_content = MagicMock()
    mock_model.generate_content_async = MagicMock()
    
    with pytest.warns(DeprecationWarning, match="'_async' is deprecated. Use 'async_client' instead."):
        client = from_vertexai(
            mock_model, 
            _async=True
        )

@patch('instructor.client_vertexai.isinstance', return_value=True)
def test_deprecated_use_async_warning(_):
    """Test that using use_async parameter raises a deprecation warning."""
    mock_model = MagicMock()
    mock_model.generate_content = MagicMock()
    mock_model.generate_content_async = MagicMock()
    
    with pytest.warns(DeprecationWarning, match="'use_async' is deprecated. Use 'async_client' instead."):
        client = from_vertexai(
            mock_model, 
            use_async=True
        )

@patch('instructor.client_vertexai.isinstance', return_value=True)
def test_multiple_async_params_error(_):
    """Test that providing multiple async parameters raises an error."""
    mock_model = MagicMock()
    mock_model.generate_content = MagicMock()
    mock_model.generate_content_async = MagicMock()
    
    with pytest.raises(ConfigurationError, match="Cannot provide multiple async parameters. Use 'async_client' instead."):
        client = from_vertexai(
            mock_model, 
            _async=True,
            use_async=True
        )
    
    with pytest.raises(ConfigurationError, match="Cannot provide multiple async parameters. Use 'async_client' instead."):
        client = from_vertexai(
            mock_model, 
            _async=True,
            async_client=True
        )
    
    with pytest.raises(ConfigurationError, match="Cannot provide multiple async parameters. Use 'async_client' instead."):
        client = from_vertexai(
            mock_model, 
            use_async=True,
            async_client=True
        )
