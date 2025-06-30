# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload
import warnings

import google.generativeai as genai

import instructor


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[True] = True,
    async_client: Literal[True] = ...,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[False] = False,
    async_client: Literal[False] = ...,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = ...,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = ...,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool | None = None,
    async_client: bool | None = None,
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

    # Handle backwards compatibility
    if use_async is not None and async_client is not None:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            "Cannot provide both 'use_async' and 'async_client'. Use 'async_client' instead."
        )

    # Determine the actual async value
    is_async = False
    
    if async_client is not None:
        is_async = async_client
    elif use_async is not None:
        warnings.warn(
            "'use_async' is deprecated. Use 'async_client' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        is_async = use_async

    if is_async:
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
