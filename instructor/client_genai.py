# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload, TypeVar

from instructor.utils import (
    convert_to_genai_messages,
    extract_genai_system_message,
    map_to_gemini_function_schema,
    update_genai_kwargs,
)

T = TypeVar("T")
from google.genai import Client

import instructor


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.GENAI_TOOLS,
        instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="GenAI", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, Client):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of google.genai.Client. "
            f"Got: {type(client).__name__}"
        )

    if use_async:

        async def async_wrapper(*args: Any, **kwargs: Any):  # type:ignore
            if kwargs.pop("stream", False):
                return await client.aio.models.generate_content_stream(*args, **kwargs)  # type:ignore
            return await client.aio.models.generate_content(*args, **kwargs)  # type:ignore

        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.GENAI,
            mode=mode,
            **kwargs,
        )

    def sync_wrapper(*args: Any, **kwargs: Any):  # type:ignore
        if kwargs.pop("stream", False):
            return client.models.generate_content_stream(*args, **kwargs)  # type:ignore

        return client.models.generate_content(*args, **kwargs)  # type:ignore

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=sync_wrapper, mode=mode),
        provider=instructor.Provider.GENAI,
        mode=mode,
        **kwargs,
    )


def handle_genai_structured_outputs(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    from google.genai import types

    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    # We validate that the schema doesn't contain any Union fields
    map_to_gemini_function_schema(response_model.model_json_schema())

    base_config = {
        "system_instruction": system_message,
        "response_mime_type": "application/json",
        "response_schema": response_model,
    }

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)

    return response_model, new_kwargs


def handle_genai_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    from google.genai import types

    schema = map_to_gemini_function_schema(response_model.model_json_schema())
    function_definition = types.FunctionDeclaration(
        name=response_model.__name__,
        description=response_model.__doc__,
        parameters=schema,
    )

    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    base_config = {
        "system_instruction": system_message,
        "tools": [types.Tool(function_declarations=[function_definition])],
        "tool_config": types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[response_model.__name__]
            ),
        ),
    }

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)

    return response_model, new_kwargs
