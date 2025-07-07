from __future__ import annotations  # type: ignore

from typing import Any, overload, TypeVar

T = TypeVar("T")

import instructor
from instructor.client import AsyncInstructor, Instructor


from cerebras.cloud.sdk import Cerebras, AsyncCerebras


@overload
def from_cerebras(
    client: Cerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cerebras(
    client: AsyncCerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cerebras(
    client: Cerebras | AsyncCerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    valid_modes = {
        instructor.Mode.CEREBRAS_TOOLS,
        instructor.Mode.CEREBRAS_JSON,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Cerebras",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, (Cerebras, AsyncCerebras)):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of Cerebras or AsyncCerebras. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, AsyncCerebras):
        create = client.chat.completions.create
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.CEREBRAS,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.CEREBRAS,
        mode=mode,
        **kwargs,
    )


def handle_cerebras_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if new_kwargs.get("stream", False):
        raise ValueError("Stream is not supported for Cerebras Tool Calling")
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model.openai_schema["name"]},
    }
    return response_model, new_kwargs


def handle_cerebras_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    instruction = f"""
You are a helpful assistant that excels at following instructions.Your task is to understand the content and provide the parsed objects in json that match the following json_schema:\n
Here is the relevant JSON schema to adhere to

<schema>
{response_model.model_json_schema()}
</schema>

Your response should consist only of a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""

    new_kwargs["messages"] = [{"role": "system", "content": instruction}] + new_kwargs[
        "messages"
    ]
    return response_model, new_kwargs
