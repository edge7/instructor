"""Tests for xAI provider implementation"""

import pytest
from unittest.mock import Mock, patch
import instructor
from instructor import Mode
from instructor.exceptions import ModeError, ClientError, ConfigurationError


class TestXAIProvider:
    """Test cases for xAI provider implementation"""
    
    def test_from_xai_with_tools_mode(self):
        """Test creating an xAI client with tools mode"""
        # Create a mock OpenAI client
        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = Mock()
        
        # Create instructor client
        client = instructor.from_xai(mock_client, mode=Mode.XAI_TOOLS)
        
        assert client.provider == instructor.Provider.XAI
        assert client.mode == Mode.XAI_TOOLS
        assert isinstance(client, instructor.Instructor)
    
    def test_from_xai_with_json_mode(self):
        """Test creating an xAI client with JSON mode"""
        # Create a mock OpenAI client
        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = Mock()
        
        # Create instructor client
        client = instructor.from_xai(mock_client, mode=Mode.XAI_JSON)
        
        assert client.provider == instructor.Provider.XAI
        assert client.mode == Mode.XAI_JSON
        assert isinstance(client, instructor.Instructor)
    
    def test_from_xai_invalid_mode(self):
        """Test that invalid modes raise an error"""
        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = Mock()
        
        with pytest.raises(ModeError) as excinfo:
            instructor.from_xai(mock_client, mode=Mode.ANTHROPIC_TOOLS)
        
        assert "xAI" in str(excinfo.value)
        assert "valid_modes" in str(excinfo.value)
    
    def test_from_xai_invalid_client(self):
        """Test that invalid clients raise an error"""
        # Client without chat attribute
        mock_client = Mock(spec=[])
        
        with pytest.raises(ClientError) as excinfo:
            instructor.from_xai(mock_client)
        
        assert "OpenAI-compatible client" in str(excinfo.value)
    
    def test_from_xai_async_client(self):
        """Test creating an async xAI client"""
        # Create a mock async OpenAI client
        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = Mock()
        mock_client._client = Mock()
        mock_client._client.is_closed = Mock(return_value=False)
        
        # Create instructor client
        client = instructor.from_xai(mock_client)
        
        assert client.provider == instructor.Provider.XAI
        assert isinstance(client, instructor.AsyncInstructor)
    
    @patch('instructor.auto_client.openai')
    def test_from_provider_xai(self, mock_openai):
        """Test using from_provider with xAI"""
        # Mock OpenAI module and client
        mock_openai_client = Mock()
        mock_openai_client.chat = Mock()
        mock_openai_client.chat.completions = Mock()
        mock_openai_client.chat.completions.create = Mock()
        mock_openai.OpenAI.return_value = mock_openai_client
        
        # Mock environment variable
        with patch('os.environ.get', return_value='test-api-key'):
            client = instructor.from_provider("xai/grok-beta")
        
        # Verify OpenAI client was created with correct parameters
        mock_openai.OpenAI.assert_called_once_with(
            api_key='test-api-key',
            base_url='https://api.x.ai/v1'
        )
        
        assert client.provider == instructor.Provider.XAI
        assert client.mode == Mode.XAI_TOOLS  # Default mode
    
    @patch('instructor.auto_client.openai')
    def test_from_provider_xai_no_api_key(self, mock_openai):
        """Test that missing API key raises an error"""
        # Mock environment to return None for API key
        with patch('os.environ.get', return_value=None):
            with pytest.raises(ConfigurationError) as excinfo:
                instructor.from_provider("xai/grok-beta")
        
        assert "XAI_API_KEY is not set" in str(excinfo.value)
    
    def test_provider_detection(self):
        """Test that xAI URLs are correctly detected"""
        from instructor.utils import get_provider, Provider
        
        assert get_provider("https://api.x.ai/v1") == Provider.XAI
        assert get_provider("https://api.xai.com/v1") == Provider.XAI
        assert get_provider("xai.com") == Provider.XAI