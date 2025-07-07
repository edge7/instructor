# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload, TypeVar
import json
from textwrap import dedent

T = TypeVar("T")

import google.generativeai as genai

import instructor


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.GEMINI_JSON,
        instructor.Mode.GEMINI_TOOLS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Gemini", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, genai.GenerativeModel):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of genai.GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    if use_async:
        create = client.generate_content_async
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.GEMINI,
            mode=mode,
            **kwargs,
        )

    create = client.generate_content
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.GEMINI,
        mode=mode,
        **kwargs,
    )


def handle_gemini_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if "model" in new_kwargs:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
        )

    from .utils import update_gemini_kwargs

    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n
        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    if new_kwargs["messages"][0]["role"] != "system":
        new_kwargs["messages"].insert(0, {"role": "system", "content": message})
    else:
        new_kwargs["messages"][0]["content"] += f"\n\n{message}"

    new_kwargs["generation_config"] = new_kwargs.get("generation_config", {}) | {
        "response_mime_type": "application/json"
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_gemini_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if "model" in new_kwargs:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
        )

    from .utils import update_gemini_kwargs

    new_kwargs["tools"] = [response_model.gemini_schema]
    new_kwargs["tool_config"] = {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [response_model.__name__],
        },
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs
