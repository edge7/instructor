from __future__ import annotations

from typing import Any, overload, TypeVar

import instructor
from instructor.client import AsyncInstructor, Instructor
from instructor.mode import Mode

from fireworks.client import Fireworks, AsyncFireworks  # type:ignore

T = TypeVar("T")


def handle_fireworks_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    # Import here to avoid circular imports
    from instructor.process_response import handle_tools

    if "stream" not in new_kwargs:
        new_kwargs["stream"] = False

    return handle_tools(response_model, new_kwargs)


def handle_fireworks_json(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if "stream" not in new_kwargs:
        new_kwargs["stream"] = False

    new_kwargs["response_format"] = {
        "type": "json_object",
        "schema": response_model.model_json_schema(),
    }
    return response_model, new_kwargs


# Mode handlers mapping
mode_handlers = {
    Mode.FIREWORKS_TOOLS: handle_fireworks_tools,
    Mode.FIREWORKS_JSON: handle_fireworks_json,
}


@overload
def from_fireworks(
    client: Fireworks,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_fireworks(
    client: AsyncFireworks,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_fireworks(
    client: Fireworks | AsyncFireworks,
    mode: instructor.Mode = instructor.Mode.FIREWORKS_JSON,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    valid_modes = {
        instructor.Mode.FIREWORKS_TOOLS,
        instructor.Mode.FIREWORKS_JSON,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Fireworks",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, (AsyncFireworks, Fireworks)):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of Fireworks or AsyncFireworks. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, AsyncFireworks):

        async def async_wrapper(*args: Any, **kwargs: Any):  # type:ignore
            if "stream" in kwargs and kwargs["stream"] is True:
                return client.chat.completions.acreate(*args, **kwargs)  # type:ignore
            return await client.chat.completions.acreate(*args, **kwargs)  # type:ignore

        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.FIREWORKS,
            mode=mode,
            **kwargs,
        )

    if isinstance(client, Fireworks):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),  # type: ignore
            provider=instructor.Provider.FIREWORKS,
            mode=mode,
            **kwargs,
        )
