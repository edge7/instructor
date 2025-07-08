from __future__ import annotations

import cohere
import instructor
from typing import (
    TypeVar,
    overload,
)
from typing import Any
from typing_extensions import ParamSpec
from pydantic import BaseModel


T_Model = TypeVar("T_Model", bound=BaseModel)
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def handle_cohere_modes(new_kwargs: dict[str, Any]) -> tuple[None, dict[str, Any]]:
    messages = new_kwargs.pop("messages", [])
    chat_history = []
    for message in messages[:-1]:
        chat_history.append(  # type: ignore
            {
                "role": message["role"],
                "message": message["content"],
            }
        )
    new_kwargs["message"] = messages[-1]["content"]
    new_kwargs["chat_history"] = chat_history
    if "model_name" in new_kwargs and "model" not in new_kwargs:
        new_kwargs["model"] = new_kwargs.pop("model_name")
    new_kwargs.pop("strict", None)
    return None, new_kwargs


def handle_cohere_json_schema(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["response_format"] = {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
    _, new_kwargs = handle_cohere_modes(new_kwargs)

    return response_model, new_kwargs


def handle_cohere_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    _, new_kwargs = handle_cohere_modes(new_kwargs)

    instruction = f"""\
Extract a valid {response_model.__name__} object based on the chat history and the json schema below.
{response_model.model_json_schema()}
The JSON schema was obtained by running:
```python
schema = {response_model.__name__}.model_json_schema()
```

The output must be a valid JSON object that `{response_model.__name__}.model_validate_json()` can successfully parse.
"""
    new_kwargs["chat_history"] = [
        {"role": "user", "message": instruction}
    ] + new_kwargs["chat_history"]
    return response_model, new_kwargs


# Mode handlers mapping
mode_handlers = {
    instructor.Mode.COHERE_JSON_SCHEMA: handle_cohere_json_schema,
    instructor.Mode.COHERE_TOOLS: handle_cohere_tools,
}


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_JSON_SCHEMA,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
):
    valid_modes = {
        instructor.Mode.COHERE_TOOLS,
        instructor.Mode.COHERE_JSON_SCHEMA,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Cohere", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (cohere.Client, cohere.AsyncClient)):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of cohere.Client or cohere.AsyncClient. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, cohere.Client):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )

    if isinstance(client, cohere.AsyncClient):
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )
