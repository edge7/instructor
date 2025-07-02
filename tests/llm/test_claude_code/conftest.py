# conftest.py
import pytest
import os
import subprocess
from instructor.client_claude_code import ClaudeCodeClient, AsyncClaudeCodeClient


def is_claude_cli_available():
    """Check if Claude CLI is available on the system."""
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def claude_available():
    """Fixture to check if Claude CLI is available."""
    return is_claude_cli_available()


@pytest.fixture(scope="session")
def client(claude_available):
    """Fixture for sync Claude Code client."""
    if not claude_available:
        pytest.skip("Claude CLI not available")
    
    return ClaudeCodeClient()


@pytest.fixture(scope="session")
def aclient(claude_available):
    """Fixture for async Claude Code client."""
    if not claude_available:
        pytest.skip("Claude CLI not available")
    
    return AsyncClaudeCodeClient()


@pytest.fixture(scope="session")
def mock_client():
    """Fixture for mock Claude Code client for testing without CLI."""
    # Create a mock client that doesn't actually execute CLI commands
    # This is useful for unit tests that don't require actual CLI execution
    from unittest.mock import Mock, MagicMock
    
    mock_client = Mock(spec=ClaudeCodeClient)
    
    # Mock the create method to return a predictable response
    def mock_create(messages, response_model=None, **kwargs):
        if response_model:
            # Return a mock instance of the response model
            if hasattr(response_model, '__name__') and response_model.__name__ == 'User':
                # For User model, return a mock user
                return response_model(name="John", age=25)
            else:
                # For other models, create a basic instance
                return response_model()
        else:
            return "Mock response text"
    
    mock_client.create = MagicMock(side_effect=mock_create)
    
    return mock_client


@pytest.fixture(scope="session")
def mock_aclient():
    """Fixture for mock async Claude Code client."""
    from unittest.mock import AsyncMock, MagicMock
    
    mock_client = MagicMock(spec=AsyncClaudeCodeClient)
    
    # Mock the create method to return a predictable response
    async def mock_create(messages, response_model=None, **kwargs):
        if response_model:
            # Return a mock instance of the response model
            if hasattr(response_model, '__name__') and response_model.__name__ == 'User':
                # For User model, return a mock user
                return response_model(name="John", age=25)
            else:
                # For other models, create a basic instance
                return response_model()
        else:
            return "Mock response text"
    
    mock_client.create = AsyncMock(side_effect=mock_create)
    
    return mock_client