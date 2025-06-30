from __future__ import annotations  # type: ignore

from typing import Any, Literal, overload

import boto3
from botocore.client import BaseClient

import instructor
from instructor.client import AsyncInstructor, Instructor


@overload  # type: ignore
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    async_client: Literal[False] = False,
    *,
    _async: Literal[False] | None = None,  # Deprecated
    **kwargs: Any,
) -> Instructor: ...


@overload  # type: ignore
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    async_client: Literal[True] = True,
    *,
    _async: Literal[True] | None = None,  # Deprecated
    **kwargs: Any,
) -> AsyncInstructor: ...


def handle_bedrock_json(
    response_model: Any,
    new_kwargs: Any,
) -> tuple[Any, Any]:
    print(f"handle_bedrock_json: response_model {response_model}")
    print(f"handle_bedrock_json: new_kwargs {new_kwargs}")
    return response_model, new_kwargs


def from_bedrock(
    client: BaseClient,
    mode: instructor.Mode = instructor.Mode.BEDROCK_JSON,
    async_client: bool | None = None,
    _async: bool | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    valid_modes = {
        instructor.Mode.BEDROCK_TOOLS,
        instructor.Mode.BEDROCK_JSON,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Bedrock",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, BaseClient):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of boto3.client (BaseClient). "
            f"Got: {type(client).__name__}"
        )

    # Determine async flag
    if async_client is not None and _async is not None:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            "Provide only the 'async_client' parameter. '_async' is deprecated."
        )

    if _async is not None:
        import warnings

        warnings.warn(
            "'_async' is deprecated and will be removed in a future release. "
            "Use 'async_client' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        async_client = _async if async_client is None else async_client

    is_async = bool(async_client)

    async def async_wrapper(**kwargs: Any):
        return client.converse(**kwargs)

    create = client.converse

    if is_async:
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.BEDROCK,
            mode=mode,
            **kwargs,
        )
    else:
        return Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.BEDROCK,
            mode=mode,
            **kwargs,
        )
