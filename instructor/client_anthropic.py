from __future__ import annotations

import anthropic
import instructor
import json
from textwrap import dedent
from typing import overload, Any, TypeVar
from instructor.utils import combine_system_messages, extract_system_messages

T = TypeVar("T")


@overload
def from_anthropic(
    client: (
        anthropic.Anthropic | anthropic.AnthropicBedrock | anthropic.AnthropicVertex
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    beta: bool = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_anthropic(
    client: (
        anthropic.AsyncAnthropic
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    beta: bool = False,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_anthropic(
    client: (
        anthropic.Anthropic
        | anthropic.AsyncAnthropic
        | anthropic.AnthropicBedrock
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
        | anthropic.AnthropicVertex
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    beta: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor instance from an Anthropic client.

    Args:
        client: An instance of Anthropic client (sync or async)
        mode: The mode to use for the client (ANTHROPIC_JSON or ANTHROPIC_TOOLS)
        beta: Whether to use beta API features (uses client.beta.messages.create)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not one of the valid Anthropic modes
        ClientError: If client is not a valid Anthropic client instance
    """
    valid_modes = {
        instructor.Mode.ANTHROPIC_JSON,
        instructor.Mode.ANTHROPIC_TOOLS,
        instructor.Mode.ANTHROPIC_REASONING_TOOLS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Anthropic",
            valid_modes=[str(m) for m in valid_modes],
        )

    valid_client_types = (
        anthropic.Anthropic,
        anthropic.AsyncAnthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
        anthropic.AsyncAnthropicVertex,
    )

    if not isinstance(client, valid_client_types):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    if beta:
        create = client.beta.messages.create
    else:
        create = client.messages.create

    if isinstance(
        client,
        (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex),
    ):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )


def handle_anthropic_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    tool_descriptions = response_model.anthropic_schema
    new_kwargs["tools"] = [tool_descriptions]
    new_kwargs["tool_choice"] = {
        "type": "tool",
        "name": response_model.__name__,
    }

    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    return response_model, new_kwargs


def handle_anthropic_reasoning_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#forcing-tool-use

    response_model, new_kwargs = handle_anthropic_tools(response_model, new_kwargs)

    # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#forcing-tool-use
    # Reasoning does not allow forced tool use
    new_kwargs["tool_choice"] = {"type": "auto"}

    # But add a message recommending only to use the tools if they are relevant
    implict_forced_tool_message = dedent(
        f"""
        Return only the tool call and no additional text.
        """
    )
    new_kwargs["system"] = combine_system_messages(
        new_kwargs.get("system"),
        [{"type": "text", "text": implict_forced_tool_message}],
    )
    return response_model, new_kwargs


def handle_anthropic_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    system_messages = extract_system_messages(new_kwargs.get("messages", []))

    if system_messages:
        new_kwargs["system"] = combine_system_messages(
            new_kwargs.get("system"), system_messages
        )

    new_kwargs["messages"] = [
        m for m in new_kwargs.get("messages", []) if m["role"] != "system"
    ]

    json_schema_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n
        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}
        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    new_kwargs["system"] = combine_system_messages(
        new_kwargs.get("system"),
        [{"type": "text", "text": json_schema_message}],
    )

    return response_model, new_kwargs
