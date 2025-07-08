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
import importlib.util
import pkg_resources

from ...mode import Mode
from ...dsl.partial import Partial
from ...hooks import Hooks

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class BaseProvider(ABC):
    """Base class for all providers.

    This class defines the interface that all providers must implement.
    It includes methods for request preparation, response processing,
    and error handling.
    """

    name: str
    supported_modes: ClassVar[set[Mode]]
    required_packages: ClassVar[dict[str, str]] = {}

    def __init__(self) -> None:
        """Initialize base provider and verify package requirements."""
        self._retry_config: dict[str, Any] = {
            "max_retries": 3,
            "timeout": 60,
            "conditions": set(),
        }
        
        # Check package requirements
        for package_name, min_version in self.required_packages.items():
            # Check if package is installed
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                raise ImportError(
                    f"Package '{package_name}' is required but not installed. "
                    f"Please install it with: pip install {package_name}"
                    + (f">={min_version}" if min_version else "")
                )

            # Check version if specified
            if min_version:
                installed_version = pkg_resources.get_distribution(package_name).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                    raise ImportError(
                        f"Package '{package_name}' version {min_version} or higher is required, "
                        f"but version {installed_version} is installed. "
                        f"Please upgrade with: pip install {package_name}>={min_version}"
                    )

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
        retry_config: int | Retrying | AsyncRetrying = 3,
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
            retry_config: Number of retries or a Retrying/AsyncRetrying instance
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
    def create_iterable(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        retry_config: int | Retrying | AsyncRetrying = 3,
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
            retry_config: Number of retries or a Retrying/AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding list items
        """
        pass
