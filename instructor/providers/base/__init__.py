"""Base provider implementation."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Set,
    TypeVar,
    Union,
)
from collections.abc import Generator, AsyncGenerator
from collections.abc import Iterable, Awaitable
from tenacity import Retrying, AsyncRetrying
from pydantic import BaseModel

from ...mode import Mode
from ...dsl.partial import Partial
from ...hooks import Hooks

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class BaseProvider(ABC):
    """Base class for all providers.

    This class defines the interface that all providers must implement.
    It includes methods for request preparation, response processing,
    error handling, and retry configuration.
    """

    name: str
    supported_modes: ClassVar[set[Mode]]

    def __init__(self) -> None:
        """Initialize base provider."""
        self._retry_config: dict[str, Any] = {
            "max_retries": 3,
            "timeout": 60,
            "conditions": set(),
        }

    @abstractmethod
    def prepare_request(
        self, response_model: Any, mode: Mode, **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare request for provider API.

        Args:
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional request options

        Returns:
            Prepared request parameters
        """
        pass

    @abstractmethod
    def process_response(
        self, response: Any, response_model: Any, mode: Mode, **kwargs: Any
    ) -> Any:
        """Process response from provider API.

        Args:
            response: Raw API response
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional processing options

        Returns:
            Processed response
        """
        pass

    @abstractmethod
    def handle_error(self, error: Exception, response: Any, **kwargs: Any) -> None:
        """Handle provider API errors.

        Args:
            error: The error that occurred
            response: Raw API response
            **kwargs: Additional error handling options
        """
        pass

    @abstractmethod
    def create(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T] | None = None,
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> T | Any | Awaitable[T] | Awaitable[Any]:
        """Create a completion and return the processed response.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Maximum number of retries or Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Processed response matching response_model type
        """
        pass

    @abstractmethod
    def create_with_completion(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any] | Awaitable[tuple[T, Any]]:
        """Create a completion and return both the processed response and raw completion.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Maximum number of retries or Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Tuple of (processed_response, raw_response)
        """
        pass

    @abstractmethod
    def create_partial(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        """Create a streaming completion that yields partial results.

        Args:
            response_model: Expected response model type
            messages: List of chat messages
            max_retries: Maximum number of retries or Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding partial results
        """
        pass

    @abstractmethod
    def create_iterable(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        """Create a streaming completion that yields list items.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Maximum number of retries or Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding list items
        """
        pass

    def configure_retry(
        self, max_retries: Optional[int] = None, timeout: Optional[int] = None
    ) -> None:
        """Configure retry settings.

        Args:
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds between retries
        """
        if max_retries is not None:
            self._retry_config["max_retries"] = max_retries
        if timeout is not None:
            self._retry_config["timeout"] = timeout

    def get_retry_conditions(self) -> set[Any]:
        """Get provider-specific retry conditions.

        Returns:
            Set of conditions that should trigger a retry
        """
        return self._retry_config["conditions"]

    def add_retry_condition(self, condition: Any) -> None:
        """Add a retry condition.

        Args:
            condition: Condition that should trigger a retry
        """
        self._retry_config["conditions"].add(condition)

    def remove_retry_condition(self, condition: Any) -> None:
        """Remove a retry condition.

        Args:
            condition: Condition to remove from retry triggers
        """
        self._retry_config["conditions"].discard(condition)

    @property
    def retry_config(self) -> dict[str, Any]:
        """Get current retry configuration.

        Returns:
            Dictionary containing retry settings
        """
        return self._retry_config.copy()
