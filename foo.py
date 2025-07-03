"""Example module used by ClaudeCodeProvider demo.

This file exists so that the prompt in `examples/claude_code_provider_demo.py`
can legitimately refer to `foo.py`.
"""

def hello(name: str = "world") -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"