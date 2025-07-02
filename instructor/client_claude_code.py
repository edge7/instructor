from __future__ import annotations

import subprocess
import json
import asyncio
import tempfile
import os
from typing import Any, overload, Union, Dict, List, Optional
from pathlib import Path

import instructor
from instructor.utils import Provider


class ClaudeCodeClient:
    """A client wrapper for Claude Code CLI."""
    
    def __init__(
        self,
        cli_path: str = "claude",
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Claude Code client.
        
        Args:
            cli_path: Path to the claude CLI binary (default: "claude")
            model: Model to use (e.g., "claude-3-5-sonnet-20241022")
            **kwargs: Additional CLI arguments
        """
        self.cli_path = cli_path
        self.model = model
        self.cli_kwargs = kwargs
        
        # Verify CLI is available
        self._verify_cli_available()
    
    def _verify_cli_available(self) -> None:
        """Verify that Claude Code CLI is available."""
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI not found or not working: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude CLI not found at '{self.cli_path}'. "
                "Please ensure Claude Code CLI is installed and available in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI verification timed out.")
    
    def _build_cli_command(self, messages: List[Dict[str, Any]], **kwargs: Any) -> List[str]:
        """Build the CLI command from messages and options."""
        cmd = [self.cli_path]
        
        # Add model if specified
        if self.model:
            cmd.extend(["--model", self.model])
        
        # Add any additional CLI kwargs
        for key, value in {**self.cli_kwargs, **kwargs}.items():
            if key == "max_tokens":
                cmd.extend(["--max-tokens", str(value)])
            elif key == "temperature":
                cmd.extend(["--temperature", str(value)])
            elif isinstance(value, bool) and value:
                cmd.append(f"--{key.replace('_', '-')}")
            elif value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def _format_messages_for_cli(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for CLI input."""
        # For Claude Code CLI, we'll combine all messages into a single prompt
        # System messages become instructions, user/assistant messages become conversation
        
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def create(
        self,
        messages: List[Dict[str, Any]],
        response_model: Optional[type] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a completion using Claude Code CLI."""
        # Format the prompt
        prompt = self._format_messages_for_cli(messages)
        
        # If we have a response model, add JSON schema instruction
        if response_model:
            schema = response_model.model_json_schema()
            prompt += f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(schema, indent=2)}"
        
        # Build and execute CLI command
        cmd = self._build_cli_command(messages, **kwargs)
        
        try:
            # Use stdin to pass the prompt
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI failed: {result.stderr}")
            
            response_text = result.stdout.strip()
            
            # If we have a response model, parse JSON
            if response_model:
                try:
                    # Try to extract JSON from the response
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed_data = json.loads(json_str)
                        return response_model(**parsed_data)
                    else:
                        # If no JSON found, try to parse the whole response
                        parsed_data = json.loads(response_text)
                        return response_model(**parsed_data)
                except (json.JSONDecodeError, TypeError) as e:
                    raise ValueError(f"Failed to parse response as JSON: {e}\nResponse: {response_text}")
            
            return response_text
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI request timed out")
        except Exception as e:
            raise RuntimeError(f"Claude CLI execution failed: {e}")


class AsyncClaudeCodeClient(ClaudeCodeClient):
    """Async version of Claude Code CLI client."""
    
    async def create(
        self,
        messages: List[Dict[str, Any]],
        response_model: Optional[type] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a completion using Claude Code CLI asynchronously."""
        # Format the prompt
        prompt = self._format_messages_for_cli(messages)
        
        # If we have a response model, add JSON schema instruction
        if response_model:
            schema = response_model.model_json_schema()
            prompt += f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(schema, indent=2)}"
        
        # Build CLI command
        cmd = self._build_cli_command(messages, **kwargs)
        
        try:
            # Use asyncio subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()),
                timeout=120  # 2 minute timeout
            )
            
            if process.returncode != 0:
                raise RuntimeError(f"Claude CLI failed: {stderr.decode()}")
            
            response_text = stdout.decode().strip()
            
            # If we have a response model, parse JSON
            if response_model:
                try:
                    # Try to extract JSON from the response
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed_data = json.loads(json_str)
                        return response_model(**parsed_data)
                    else:
                        # If no JSON found, try to parse the whole response
                        parsed_data = json.loads(response_text)
                        return response_model(**parsed_data)
                except (json.JSONDecodeError, TypeError) as e:
                    raise ValueError(f"Failed to parse response as JSON: {e}\nResponse: {response_text}")
            
            return response_text
            
        except asyncio.TimeoutError:
            raise RuntimeError("Claude CLI request timed out")
        except Exception as e:
            raise RuntimeError(f"Claude CLI execution failed: {e}")


@overload
def from_claude_code(
    cli_path: str = "claude",
    model: Optional[str] = None,
    async_client: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    ...


@overload 
def from_claude_code(
    cli_path: str = "claude", 
    model: Optional[str] = None,
    async_client: bool = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor:
    ...


def from_claude_code(
    cli_path: str = "claude",
    model: Optional[str] = None,
    async_client: bool = False,
    **kwargs: Any,
) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
    """Create an Instructor instance from Claude Code CLI.

    Args:
        cli_path: Path to the claude CLI binary (default: "claude")
        model: Model to use (e.g., "claude-3-5-sonnet-20241022")
        async_client: Whether to return an async client
        **kwargs: Additional keyword arguments to pass to the CLI

    Returns:
        An Instructor instance (sync or async depending on async_client)

    Raises:
        RuntimeError: If Claude Code CLI is not available or not working
        
    Examples:
        >>> import instructor
        >>> from instructor import from_claude_code
        >>> 
        >>> # Sync client
        >>> client = from_claude_code()
        >>> 
        >>> # Async client
        >>> async_client = from_claude_code(async_client=True)
        >>> 
        >>> # With specific model
        >>> client = from_claude_code(model="claude-3-5-sonnet-20241022")
    """
    if async_client:
        claude_client = AsyncClaudeCodeClient(cli_path=cli_path, model=model, **kwargs)
        return instructor.AsyncInstructor(
            client=claude_client,
            create=instructor.patch(create=claude_client.create, mode=instructor.Mode.CLAUDE_CODE_JSON),
            provider=Provider.CLAUDE_CODE,
            mode=instructor.Mode.CLAUDE_CODE_JSON,
            **kwargs,
        )
    else:
        claude_client = ClaudeCodeClient(cli_path=cli_path, model=model, **kwargs)
        return instructor.Instructor(
            client=claude_client,
            create=instructor.patch(create=claude_client.create, mode=instructor.Mode.CLAUDE_CODE_JSON),
            provider=Provider.CLAUDE_CODE,
            mode=instructor.Mode.CLAUDE_CODE_JSON,
            **kwargs,
        )