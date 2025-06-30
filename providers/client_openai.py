from __future__ import annotations

"""Implementation of the OpenAI provider using the new abstraction layer.

The goal is **not** to change the public API – existing user code that calls
`instructor.client.from_openai` should continue to work.  This module only
implements the new provider interface so we can gradually migrate other
providers in the future.
"""

from functools import partial
from typing import Any

# The provider base classes live in the same top-level *providers* package.
from providers.base import BaseProvider, register_provider

# We keep all heavy imports under a guard so that users who do not work with
# OpenAI are not forced to install the dependency.
try:
    import openai  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    openai = None  # type: ignore  # pylint: disable=invalid-name

import instructor
from instructor.utils import Provider, get_provider
from instructor.client import (
    Instructor,
    AsyncInstructor,
    map_chat_completion_to_response,
    async_map_chat_completion_to_response,
)


@register_provider
class OpenAIProvider(BaseProvider):
    """Concrete provider for the `openai` Python SDK."""

    name = "openai"

    # NOTE: The return type is intentionally ``Any`` to keep *providers* free of
    # heavy instructor imports.  Callers (or static type checkers) know the
    # exact return value.
    @classmethod
    def from_client(
        cls,  # noqa: D401 – descriptor method, not a property
        client: Any,
        *,
        mode: instructor.Mode = instructor.Mode.TOOLS,
        **kwargs: Any,
    ) -> Instructor | AsyncInstructor:  # type: ignore[override]
        if openai is None:
            raise ImportError(
                "The 'openai' package is required to use the OpenAI provider."
            )

        if hasattr(client, "base_url"):
            provider = get_provider(str(client.base_url))
        else:
            provider = Provider.OPENAI

        if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
            import warnings

            warnings.warn(
                "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. "
                "Unexpected behavior may occur with other client types.",
                stacklevel=2,
            )

        # ---------------------------------------------------------------------
        # Validate mode based on provider – these checks mirror the original
        # implementation to maintain exact behaviour.
        # ---------------------------------------------------------------------
        if provider in {Provider.OPENROUTER}:
            assert mode in {
                instructor.Mode.TOOLS,
                instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS,
                instructor.Mode.JSON,
            }

        if provider in {Provider.ANYSCALE, Provider.TOGETHER}:
            assert mode in {
                instructor.Mode.TOOLS,
                instructor.Mode.JSON,
                instructor.Mode.JSON_SCHEMA,
                instructor.Mode.MD_JSON,
            }

        if provider in {Provider.OPENAI, Provider.DATABRICKS}:
            assert mode in {
                instructor.Mode.TOOLS,
                instructor.Mode.JSON,
                instructor.Mode.FUNCTIONS,
                instructor.Mode.PARALLEL_TOOLS,
                instructor.Mode.MD_JSON,
                instructor.Mode.TOOLS_STRICT,
                instructor.Mode.JSON_O1,
                instructor.Mode.RESPONSES_TOOLS,
                instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            }

        # ------------------------------------------------------------------
        # Build the *create* callable passed down to *Instructor*.
        # ------------------------------------------------------------------
        if isinstance(client, openai.OpenAI):
            create_fn = client.chat.completions.create
            if mode in {
                instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                instructor.Mode.RESPONSES_TOOLS,
            }:
                create_fn = partial(map_chat_completion_to_response, client=client)

            return Instructor(
                client=client,
                create=instructor.patch(create=create_fn, mode=mode),
                mode=mode,
                provider=provider,
                **kwargs,
            )

        # Async version ------------------------------------------------------
        create_fn = client.chat.completions.create  # type: ignore[attr-defined]
        if mode in {
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            instructor.Mode.RESPONSES_TOOLS,
        }:
            create_fn = partial(  # type: ignore[misc]
                async_map_chat_completion_to_response, client=client
            )

        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=create_fn, mode=mode),
            mode=mode,
            provider=provider,
            **kwargs,
        )