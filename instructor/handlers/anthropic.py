from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.utils import combine_system_messages, extract_system_messages

from .base import ResponseHandler
from .registry import register

T = TypeVar("T")


class AnthropicHandler(ResponseHandler):
    """Request-side handler for Anthropic modes (tools & JSON)."""

    supported_modes = (
        Mode.ANTHROPIC_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
        Mode.ANTHROPIC_JSON,
    )

    # Main public API -----------------------------------------------------
    def prepare_request(
        self,
        mode: Mode,
        response_model: type[T] | None,
        call_kwargs: dict[str, Any],
    ) -> tuple[type[T] | None, dict[str, Any]]:
        if response_model is None:
            # Nothing to patch if user did not request structured output
            return None, call_kwargs

        # Work on a shallow copy so we never mutate caller's dict
        new_kwargs = call_kwargs.copy()

        if mode is Mode.ANTHROPIC_TOOLS:
            return self._anthropic_tools(response_model, new_kwargs)
        elif mode is Mode.ANTHROPIC_REASONING_TOOLS:
            return self._anthropic_reasoning_tools(response_model, new_kwargs)
        elif mode is Mode.ANTHROPIC_JSON:
            return self._anthropic_json(response_model, new_kwargs)
        else:  # pragma: no cover — defensive fallback
            return response_model, new_kwargs

    # ------------------------------------------------------------------
    # Internal helpers (largely copied from previous implementation)
    # ------------------------------------------------------------------
    def _anthropic_tools(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        tool_descriptions = response_model.anthropic_schema  # type: ignore[attr-defined]
        new_kwargs["tools"] = [tool_descriptions]
        new_kwargs["tool_choice"] = {"type": "tool", "name": response_model.__name__}

        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )

        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m.get("role") != "system"
        ]
        return response_model, new_kwargs

    def _anthropic_reasoning_tools(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        # Start from baseline tool handling
        response_model, new_kwargs = self._anthropic_tools(response_model, new_kwargs)

        # Reasoning mode cannot force tool usage
        new_kwargs["tool_choice"] = {"type": "auto"}

        implicit_message = dedent(
            """
            Return only the tool call and no additional text.
            """
        )
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), [{"type": "text", "text": implicit_message}]
        )
        return response_model, new_kwargs

    def _anthropic_json(
        self, response_model: type[T], new_kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        system_messages = extract_system_messages(new_kwargs.get("messages", []))
        if system_messages:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs.get("system"), system_messages
            )

        # Strip system messages from the messages list
        new_kwargs["messages"] = [
            m for m in new_kwargs.get("messages", []) if m.get("role") != "system"
        ]

        json_schema_message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n
            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}  # type: ignore[attr-defined]

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        # Anthropic uses separate system field; merge or create it
        if "system" not in new_kwargs:
            new_kwargs["system"] = [{"type": "text", "text": json_schema_message}]
        else:
            new_kwargs["system"] = combine_system_messages(
                new_kwargs["system"], [{"type": "text", "text": json_schema_message}]
            )
        return response_model, new_kwargs

# Register a singleton instance with the central registry
register(AnthropicHandler())