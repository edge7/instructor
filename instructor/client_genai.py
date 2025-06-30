# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload

from google.genai import Client

import instructor


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    async_client: Literal[True] = True,
    *,
    use_async: Literal[True] | None = None,  # Deprecated
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    async_client: Literal[False] = False,
    *,
    use_async: Literal[False] | None = None,  # Deprecated
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    async_client: bool | None = None,
    use_async: bool | None = None,
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

    # Determine async mode
    if async_client is not None and use_async is not None:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            "Provide only the 'async_client' parameter. 'use_async' is deprecated."
        )

    if use_async is not None:
        import warnings

        warnings.warn(
            "'use_async' is deprecated and will be removed in a future release. "
            "Use 'async_client' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        async_client = use_async if async_client is None else async_client

    is_async = bool(async_client)

    if is_async:
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
