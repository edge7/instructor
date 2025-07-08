"""Base provider implementation."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar, Union
from collections.abc import Generator, AsyncGenerator
from collections.abc import Iterable
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
    It includes methods for request preparation and response processing.
    """

    name: str
    supported_modes: ClassVar[set[Mode]]
    required_packages: ClassVar[dict[str, str]] = {}

    def __init__(self) -> None:
        """Initialize base provider and verify package requirements."""
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
        """Prepare request for provider API."""
        pass

    @abstractmethod
    def process_response(
        self, response: Any, response_model: Any, mode: Mode, **kwargs: Any
    ) -> Any:
        """Process response from provider API."""
        pass

    @abstractmethod
    def async_create_with_completion(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Create a completion asynchronously and return both the processed response and raw completion."""
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
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> Generator[T, None, None]:
        """Create a streaming completion that yields partial results."""
        pass

    @abstractmethod
    async def async_create_partial(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        """Create a streaming completion that yields partial results asynchronously."""
        pass
