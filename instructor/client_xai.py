from __future__ import annotations

from typing import overload, Any

import instructor


@overload
def from_xai(
    client: Any,
    mode: instructor.Mode = instructor.Mode.XAI_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_xai(
    client: Any,
    mode: instructor.Mode = instructor.Mode.XAI_TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_xai(
    client: Any,
    mode: instructor.Mode = instructor.Mode.XAI_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """
    Create an instructor client from an xAI client.
    
    Args:
        client: The xAI client instance (OpenAI-compatible client)
        mode: The mode to use (XAI_TOOLS or XAI_JSON)
        **kwargs: Additional arguments to pass to the instructor
        
    Returns:
        An instructor client configured for xAI
    """
    # Valid modes for xAI
    valid_modes = {
        instructor.Mode.XAI_JSON,
        instructor.Mode.XAI_TOOLS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="xAI", valid_modes=[str(m) for m in valid_modes]
        )

    # Import OpenAI types dynamically to avoid hard dependency
    try:
        import openai
    except ImportError:
        from instructor.exceptions import ClientError
        
        raise ClientError(
            "xAI provider requires the OpenAI SDK to be installed. "
            "Install it with: pip install openai"
        )

    # Check if it's an OpenAI-compatible client
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an OpenAI-compatible client with chat.completions.create method. "
            f"Got: {type(client).__name__}"
        )

    # Since xAI uses OpenAI-compatible API, we need to map the modes
    # to the appropriate OpenAI modes
    mode_mapping = {
        instructor.Mode.XAI_JSON: instructor.Mode.JSON,
        instructor.Mode.XAI_TOOLS: instructor.Mode.TOOLS,
    }
    
    mapped_mode = mode_mapping.get(mode, mode)
    
    # Check if it's an async client
    if hasattr(client, "_client") and hasattr(client._client, "is_closed"):
        # This is likely an async client
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mapped_mode),
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )
    else:
        # This is a sync client
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mapped_mode),
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )