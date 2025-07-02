import pytest
import instructor
from pydantic import BaseModel
from instructor.client_claude_code import ClaudeCodeClient, AsyncClaudeCodeClient
from instructor import from_claude_code


class User(BaseModel):
    name: str
    age: int


class UserProfile(BaseModel):
    name: str
    age: int
    email: str


def test_claude_code_client_initialization():
    """Test that Claude Code client can be initialized."""
    client = ClaudeCodeClient()
    assert client.cli_path == "claude"
    assert client.model is None
    
    # Test with custom parameters
    client_custom = ClaudeCodeClient(cli_path="/usr/local/bin/claude", model="claude-3-5-sonnet")
    assert client_custom.cli_path == "/usr/local/bin/claude"
    assert client_custom.model == "claude-3-5-sonnet"


def test_async_claude_code_client_initialization():
    """Test that async Claude Code client can be initialized."""
    client = AsyncClaudeCodeClient()
    assert client.cli_path == "claude"
    assert client.model is None


def test_from_claude_code_sync():
    """Test from_claude_code function with sync client."""
    client = from_claude_code()
    assert isinstance(client, instructor.Instructor)
    assert client.provider == instructor.Provider.CLAUDE_CODE
    assert client.mode == instructor.Mode.CLAUDE_CODE_JSON


def test_from_claude_code_async():
    """Test from_claude_code function with async client."""
    client = from_claude_code(async_client=True)
    assert isinstance(client, instructor.AsyncInstructor)
    assert client.provider == instructor.Provider.CLAUDE_CODE
    assert client.mode == instructor.Mode.CLAUDE_CODE_JSON


def test_from_claude_code_with_model():
    """Test from_claude_code function with specific model."""
    client = from_claude_code(model="claude-3-5-sonnet")
    assert isinstance(client, instructor.Instructor)


def test_cli_command_building():
    """Test CLI command building."""
    client = ClaudeCodeClient(model="claude-3-5-sonnet")
    
    messages = [{"role": "user", "content": "Hello"}]
    cmd = client._build_cli_command(messages)
    
    assert "claude" in cmd
    assert "--model" in cmd
    assert "claude-3-5-sonnet" in cmd


def test_message_formatting():
    """Test message formatting for CLI."""
    client = ClaudeCodeClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    formatted = client._format_messages_for_cli(messages)
    
    assert "System: You are a helpful assistant." in formatted
    assert "Human: Hello" in formatted
    assert "Assistant: Hi there!" in formatted
    assert "Human: How are you?" in formatted


@pytest.mark.skip(reason="Requires actual Claude CLI - use for integration testing")
def test_real_claude_cli_integration(client):
    """Integration test with real Claude CLI (skipped by default)."""
    instructor_client = instructor.from_claude_code()
    
    response = instructor_client.create(
        response_model=User,
        messages=[{"role": "user", "content": "John is 25 years old"}]
    )
    
    assert isinstance(response, User)
    assert response.name == "John"
    assert response.age == 25


@pytest.mark.skip(reason="Requires actual Claude CLI - use for integration testing")
@pytest.mark.asyncio
async def test_real_async_claude_cli_integration(aclient):
    """Async integration test with real Claude CLI (skipped by default)."""
    instructor_client = instructor.from_claude_code(async_client=True)
    
    response = await instructor_client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Alice is 30 years old"}]
    )
    
    assert isinstance(response, User)
    assert response.name == "Alice"
    assert response.age == 30


def test_mock_claude_integration(mock_client):
    """Test with mock client to verify interface without CLI dependency."""
    # Test that our mock works as expected
    response = mock_client.create(
        messages=[{"role": "user", "content": "John is 25 years old"}],
        response_model=User
    )
    
    assert isinstance(response, User)
    assert response.name == "John"
    assert response.age == 25


@pytest.mark.asyncio
async def test_mock_async_claude_integration(mock_aclient):
    """Test async mock client."""
    response = await mock_aclient.create(
        messages=[{"role": "user", "content": "John is 25 years old"}],
        response_model=User
    )
    
    assert isinstance(response, User)
    assert response.name == "John"
    assert response.age == 25


def test_provider_integration():
    """Test Claude Code integration via from_provider."""
    try:
        client = instructor.from_provider("claude_code/claude-3-5-sonnet")
        assert isinstance(client, instructor.Instructor)
        assert client.provider == instructor.Provider.CLAUDE_CODE
    except Exception as e:
        # This might fail if Claude CLI is not available, which is expected
        assert "Claude Code CLI" in str(e) or "not found" in str(e)


def test_error_handling_cli_not_found():
    """Test error handling when CLI is not found."""
    with pytest.raises(RuntimeError) as exc_info:
        ClaudeCodeClient(cli_path="nonexistent-claude-cli")
    
    assert "Claude CLI not found" in str(exc_info.value)


def test_json_schema_instruction_generation():
    """Test that JSON schema instructions are properly generated."""
    client = ClaudeCodeClient()
    
    # Mock the subprocess call to avoid actual CLI execution
    import unittest.mock
    
    with unittest.mock.patch('subprocess.run') as mock_run:
        # Configure mock to return successful JSON response
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"name": "John", "age": 25}'
        mock_run.return_value.stderr = ''
        
        response = client.create(
            messages=[{"role": "user", "content": "Extract user info"}],
            response_model=User
        )
        
        # Verify the call was made
        assert mock_run.called
        
        # Get the input that was passed to the CLI
        call_args = mock_run.call_args
        input_text = call_args[1]['input']  # The input parameter
        
        # Verify JSON schema instruction was added
        assert "Please respond with valid JSON that matches this schema" in input_text
        assert '"name"' in input_text  # Part of the User schema
        assert '"age"' in input_text   # Part of the User schema
        
        # Verify response was parsed correctly
        assert isinstance(response, User)
        assert response.name == "John"
        assert response.age == 25