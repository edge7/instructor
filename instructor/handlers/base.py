from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from instructor.mode import Mode


# ---------------------------------------------------------------------------
# Helper protocol to express models that expose `.model_json_schema()`
# and (optionally) `.anthropic_schema`. This lets us keep strong typing while
# avoiding `# type: ignore` comments across the codebase.
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsModelJsonSchema(Protocol):
    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...

    anthropic_schema: Any  # Provider-specific


T = TypeVar("T", bound=SupportsModelJsonSchema)


class ResponseHandler(Protocol):
    """Protocol for request/response transformation handlers."""

    supported_modes: tuple[Mode, ...]

    # Request-side ---------------------------------------------------------
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

    # Response-side --------------------------------------------------------
    def parse_response(
        self,
        mode: Mode,
        raw_completion: Any,
        *,
        response_model: type[T] | None,
        stream: bool,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any | None:
        """Convert the raw provider completion into the final return type.

        If the handler does not apply, return ``None`` to fall back to the
        legacy parsing path.
        """

        ...