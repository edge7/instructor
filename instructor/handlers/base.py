from __future__ import annotations

from typing import Any, Protocol, TypeVar

from instructor.mode import Mode

T = TypeVar("T")


class ResponseHandler(Protocol):
    """Protocol for request/response transformation handlers."""

    supported_modes: tuple[Mode, ...]

    def prepare_request(
        self,
        mode: Mode,
        response_model: type[T] | None,
        call_kwargs: dict[str, Any],
    ) -> tuple[type[T] | None, dict[str, Any]]:
        """Mutate/augment kwargs before the provider call.

        Args:
            mode: The operation mode triggering this handler.
            response_model: Parsed response model requested by the caller.
            call_kwargs: Keyword arguments destined for the provider SDK.

        Returns
        -------
        A tuple of (possibly modified) response_model and kwargs.
        """

        ...