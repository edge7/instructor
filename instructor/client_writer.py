# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations


import instructor
from writerai import AsyncWriter, Writer
from typing import overload, Any, TypeVar

T = TypeVar("T")


def handle_writer_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = "auto"
    return response_model, new_kwargs


def handle_writer_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": response_model.model_json_schema()},
    }

    return response_model, new_kwargs


# Writer mode handlers mapping
mode_handlers = {
    instructor.Mode.WRITER_TOOLS: handle_writer_tools,
    instructor.Mode.WRITER_JSON: handle_writer_json,
}


@overload
def from_writer(
    client: Writer,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_writer(
    client: AsyncWriter,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_writer(
    client: Writer | AsyncWriter,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    **kwargs: Any,
) -> instructor.client.Instructor | instructor.client.AsyncInstructor:
    valid_modes = {instructor.Mode.WRITER_TOOLS, instructor.Mode.WRITER_JSON}

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Writer", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (Writer, AsyncWriter)):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of Writer or AsyncWriter. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, Writer):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat.chat, mode=mode),
            provider=instructor.Provider.WRITER,
            mode=mode,
            **kwargs,
        )

    return instructor.AsyncInstructor(
        client=client,
        create=instructor.patch(create=client.chat.chat, mode=mode),
        provider=instructor.Provider.WRITER,
        mode=mode,
        **kwargs,
    )
