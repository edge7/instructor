from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def handle_openai_tools(
    response_model: type[T],
    new_kwargs: dict[str, Any],
    *,
    tool_choice: str | dict[str, Any] | None = None,
) -> tuple[type[T], dict[str, Any]]:
    """Generic OpenAI-style tools handler."""
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    if tool_choice is None:
        tool_choice = {
            "type": "function",
            "function": {"name": response_model.openai_schema["name"]},
        }
    new_kwargs["tool_choice"] = tool_choice
    return response_model, new_kwargs


def handle_openai_json_schema(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    """Generic JSON handler for OpenAI compatible providers."""
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": response_model.model_json_schema()},
    }
    return response_model, new_kwargs
