# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations


from mistralai import Mistral
import instructor
from typing import overload, Any, Literal
import warnings


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: Literal[True] = True,
    async_client: Literal[True] = ...,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: Literal[False] = False,
    async_client: Literal[False] = ...,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: bool = ...,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: bool = ...,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: bool | None = None,
    async_client: bool | None = None,
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
