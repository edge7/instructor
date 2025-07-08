"""Base provider implementation."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Optional,
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
from ...retry import retry_sync, retry_async

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
    ) -> T | Any:
        """Create a completion and return the processed response.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Processed response matching response_model type
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages

        def create_fn(*args: Any, **kwargs: Any) -> Any:
            return self.client.chat.completions.create(**kwargs)

        return retry_sync(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

    async def async_create(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T] | None = None,
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> T | Any:
        """Create a completion asynchronously and return the processed response.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Processed response matching response_model type
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages

        async def create_fn(*args: Any, **kwargs: Any) -> Any:
            return await self.client.chat.completions.create(**kwargs)

        return await retry_async(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

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
    ) -> tuple[T, Any]:
        """Create a completion and return both the processed response and raw completion.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Tuple of (processed_response, raw_response)
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages

        def create_fn(*args: Any, **kwargs: Any) -> Any:
            completion = self.client.chat.completions.create(**kwargs)
            result = self.process_response(
                completion,
                response_model,
                kwargs.get("mode", Mode.FUNCTIONS),
                validation_context=validation_context,
                context=context,
                strict=strict,
            )
            return result, completion

        return retry_sync(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

    async def async_create_with_completion(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Create a completion asynchronously and return both the processed response and raw completion.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Tuple of (processed_response, raw_response)
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages

        async def create_fn(*args: Any, **kwargs: Any) -> Any:
            completion = await self.client.chat.completions.create(**kwargs)
            result = self.process_response(
                completion,
                response_model,
                kwargs.get("mode", Mode.FUNCTIONS),
                validation_context=validation_context,
                context=context,
                strict=strict,
            )
            return result, completion

        return await retry_async(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

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
    ) -> Generator[T, None, None]:
        """Create a streaming completion that yields partial results.

        Args:
            response_model: Expected response model type
            messages: List of chat messages
            max_retries: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding partial results
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        def create_fn(*args: Any, **kwargs: Any) -> Generator[T, None, None]:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                result = self.process_response(
                    chunk,
                    response_model,
                    kwargs.get("mode", Mode.FUNCTIONS),
                    validation_context=validation_context,
                    context=context,
                    strict=strict,
                    partial=True,
                )
                if result is not None:
                    yield result

        return retry_sync(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

    async def async_create_partial(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        """Create a streaming completion that yields partial results asynchronously.

        Args:
            response_model: Expected response model type
            messages: List of chat messages
            max_retries: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            AsyncGenerator yielding partial results
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        async def create_fn(*args: Any, **kwargs: Any) -> AsyncGenerator[T, None]:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                result = self.process_response(
                    chunk,
                    response_model,
                    kwargs.get("mode", Mode.FUNCTIONS),
                    validation_context=validation_context,
                    context=context,
                    strict=strict,
                    partial=True,
                )
                if result is not None:
                    yield result

        return await retry_async(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

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
    ) -> Generator[T, None, None]:
        """Create a streaming completion that yields list items.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding list items
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        def create_fn(*args: Any, **kwargs: Any) -> Generator[T, None, None]:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                result = self.process_response(
                    chunk,
                    response_model,
                    kwargs.get("mode", Mode.FUNCTIONS),
                    validation_context=validation_context,
                    context=context,
                    strict=strict,
                    iterable=True,
                )
                if result is not None:
                    yield result

        return retry_sync(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )

    async def async_create_iterable(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        """Create a streaming completion that yields list items asynchronously.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            max_retries: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            **kwargs: Additional provider-specific options

        Returns:
            AsyncGenerator yielding list items
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, kwargs.get("mode", Mode.FUNCTIONS), **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        async def create_fn(*args: Any, **kwargs: Any) -> AsyncGenerator[T, None]:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                result = self.process_response(
                    chunk,
                    response_model,
                    kwargs.get("mode", Mode.FUNCTIONS),
                    validation_context=validation_context,
                    context=context,
                    strict=strict,
                    iterable=True,
                )
                if result is not None:
                    yield result

        return await retry_async(
            func=create_fn,
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=[],
            kwargs=request_params,
            strict=strict,
            mode=kwargs.get("mode", Mode.FUNCTIONS),
            hooks=hooks,
        )
