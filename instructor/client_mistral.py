# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations


from mistralai import Mistral
import instructor
from typing import overload, Any, Literal


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    async_client: Literal[True] = True,
    *,
    use_async: Literal[True] | None = None,  # Deprecated
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    async_client: Literal[False] = False,
    *,
    use_async: Literal[False] | None = None,  # Deprecated
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    async_client: bool | None = None,
    use_async: bool | None = None,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.MISTRAL_TOOLS,
        instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Mistral",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, Mistral):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of mistralai.Mistral. "
            f"Got: {type(client).__name__}"
        )

    # Determine async mode, prioritising new `async_client` flag
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

    is_async_mode = bool(async_client)

    if is_async_mode:

        async def async_wrapper(
            *args: Any, **kwargs: dict[str, Any]
        ):  # Handler for async streaming
            if kwargs.pop("stream", False):
                return await client.chat.stream_async(*args, **kwargs)
            return await client.chat.complete_async(*args, **kwargs)

        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )

    def sync_wrapper(
        *args: Any, **kwargs: dict[str, Any]
    ):  # Handler for sync streaming
        if kwargs.pop("stream", False):
            return client.chat.stream(*args, **kwargs)
        return client.chat.complete(*args, **kwargs)

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=sync_wrapper, mode=mode),
        provider=instructor.Provider.MISTRAL,
        mode=mode,
        **kwargs,
    )
