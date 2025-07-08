"""OpenAI error handling utilities."""

from typing import Any
from openai.types.chat import ChatCompletion

def reask_tools(
    kwargs: dict[str, Any],
    response: ChatCompletion,
    exception: Exception,
) -> dict[str, Any]:
    """Handle reask for tools mode.
    
    Args:
        kwargs: Request kwargs
        response: Raw API response
        exception: The error that occurred
        
    Returns:
        Updated kwargs for retry
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": (
                f"Validation Error found:\n{exception}\n"
                "Recall the function correctly, fix the errors"
            ),
        })
        
    kwargs["messages"].extend(reask_msgs)
    return kwargs

def dump_message(message: Any) -> dict[str, Any]:
    """Convert message to dict format.
    
    Args:
        message: Message to convert
        
    Returns:
        Dict representation of message
    """
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in (message.tool_calls or [])
        ] if message.tool_calls else None
    } 