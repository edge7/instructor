"""OpenAI provider implementation."""

from typing import Any, ClassVar, Optional, TypeVar, Union
from collections.abc import Generator, AsyncGenerator
from collections.abc import Iterable
from tenacity import Retrying, AsyncRetrying
from pydantic import BaseModel
import openai
from openai.types.chat import ChatCompletion
from openai._types import NotGiven, Undefined

from ..base import BaseProvider
from ...mode import Mode
from ...dsl.partial import Partial
from ...hooks import Hooks
from .response import process_openai_response
from .errors import handle_openai_error

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    name = "openai"
    supported_modes = {Mode.FUNCTIONS, Mode.TOOLS, Mode.TOOLS_STRICT, Mode.JSON}
    required_packages = {"openai": "1.0.0"}

    def __init__(
        self, client: Optional[openai.OpenAI | openai.AsyncOpenAI] = None
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            client: Optional OpenAI client instance
        """
        super().__init__()
        self.client = client or openai.OpenAI()
        
        # Configure retry conditions for OpenAI
        self._retry_handler.configure(
            conditions={
                "rate_limit_error",
                "timeout_error",
                "server_error"
            }
        )

    def prepare_request(
        self, response_model: Any, mode: Mode, **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare request for OpenAI API.

        Args:
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional request options

        Returns:
            Prepared request parameters
        """
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported by OpenAI provider")

        request_params = {
            "model": kwargs.pop("model", "gpt-3.5-turbo"),
            "temperature": kwargs.pop("temperature", 0.0),
            "max_tokens": kwargs.pop("max_tokens", None),
            "top_p": kwargs.pop("top_p", 1.0),
            "frequency_penalty": kwargs.pop("frequency_penalty", 0.0),
            "presence_penalty": kwargs.pop("presence_penalty", 0.0),
            "stop": kwargs.pop("stop", None),
            "seed": kwargs.pop("seed", None),
            "response_format": kwargs.pop("response_format", None),
        }

        # Remove None values
        request_params = {k: v for k, v in request_params.items() if v is not None}

        # Add any remaining kwargs
        request_params.update(kwargs)

        return request_params

    def process_response(
        self, response: Any, response_model: Any, mode: Mode, **kwargs: Any
    ) -> Any:
        """Process response from OpenAI API.

        Args:
            response: Raw API response
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional processing options

        Returns:
            Processed response
        """
        return process_openai_response(
            response=response,
            response_model=response_model,
            mode=mode,
            **kwargs
        )

    def handle_error(self, error: Exception, response: Any, **kwargs: Any) -> None:
        """Handle OpenAI API errors.

        Args:
            error: The error that occurred
            response: Raw API response
            **kwargs: Additional error handling options
        """
        handle_openai_error(error, response, **kwargs)

    def create(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T] | None = None,
        retry_config: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> T | Any:
        """Create a completion and return the processed response.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            retry_config: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            Processed response matching response_model type
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages

        # Create completion with retries
        for attempt in Retrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create completion
                    completion = self.client.chat.completions.create(**request_params)

                    # Run post-completion hooks
                    if hooks:
                        hooks.post_completion(completion)

                    # Process response
                    result = self.process_response(
                        completion,
                        response_model,
                        mode,
                        validation_context=validation_context,
                        context=context,
                        strict=strict,
                    )

                    return result

                except Exception as e:
                    self.handle_error(e, completion if "completion" in locals() else None)
                    raise

    async def async_create(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T] | None = None,
        retry_config: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> T | Any:
        """Create a completion asynchronously and return the processed response.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            retry_config: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            Processed response matching response_model type
        """
        if not isinstance(self.client, openai.AsyncOpenAI):
            raise TypeError("Async operations require an AsyncOpenAI client")

        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages

        # Create completion with retries
        async for attempt in AsyncRetrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create completion
                    completion = await self.client.chat.completions.create(**request_params)

                    # Run post-completion hooks
                    if hooks:
                        hooks.post_completion(completion)

                    # Process response
                    result = self.process_response(
                        completion,
                        response_model,
                        mode,
                        validation_context=validation_context,
                        context=context,
                        strict=strict,
                    )

                    return result

                except Exception as e:
                    self.handle_error(e, completion if "completion" in locals() else None)
                    raise

    def create_with_completion(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        retry_config: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> tuple[T, ChatCompletion]:
        """Create a completion and return both the processed response and raw completion.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            retry_config: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            Tuple of (processed_response, raw_response)
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages

        # Create completion with retries
        for attempt in Retrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create completion
                    completion = self.client.chat.completions.create(**request_params)

                    # Run post-completion hooks
                    if hooks:
                        hooks.post_completion(completion)

                    # Process response
                    result = self.process_response(
                        completion,
                        response_model,
                        mode,
                        validation_context=validation_context,
                        context=context,
                        strict=strict,
                    )

                    return result, completion

                except Exception as e:
                    self.handle_error(
                        e, completion if "completion" in locals() else None
                    )
                    raise

    async def async_create_with_completion(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        retry_config: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> tuple[T, ChatCompletion]:
        """Create a completion asynchronously and return both the processed response and raw completion.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            retry_config: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            Tuple of (processed_response, raw_response)
        """
        if not isinstance(self.client, openai.AsyncOpenAI):
            raise TypeError("Async operations require an AsyncOpenAI client")

        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages

        # Create completion with retries
        async for attempt in AsyncRetrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create completion
                    completion = await self.client.chat.completions.create(**request_params)

                    # Run post-completion hooks
                    if hooks:
                        hooks.post_completion(completion)

                    # Process response
                    result = self.process_response(
                        completion,
                        response_model,
                        mode,
                        validation_context=validation_context,
                        context=context,
                        strict=strict,
                    )

                    return result, completion

                except Exception as e:
                    self.handle_error(
                        e, completion if "completion" in locals() else None
                    )
                    raise

    def create_partial(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        retry_config: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> Generator[T, None, None]:
        """Create a streaming completion that yields partial results.

        Args:
            response_model: Expected response model type
            messages: List of chat messages
            retry_config: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding partial results
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        # Create streaming completion with retries
        for attempt in Retrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create streaming completion
                    stream = self.client.chat.completions.create(**request_params)

                    # Process stream chunks
                    for chunk in stream:
                        # Run post-chunk hooks
                        if hooks:
                            hooks.post_completion(chunk)

                        # Process chunk
                        result = self.process_response(
                            chunk,
                            response_model,
                            mode,
                            validation_context=validation_context,
                            context=context,
                            strict=strict,
                            partial=True,
                        )

                        if result is not None:
                            yield result

                except Exception as e:
                    self.handle_error(e, None)
                    raise

    async def async_create_partial(
        self,
        response_model: type[T],
        messages: list[dict[str, Any]],
        retry_config: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        """Create a streaming completion that yields partial results asynchronously.

        Args:
            response_model: Expected response model type
            messages: List of chat messages
            retry_config: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            AsyncGenerator yielding partial results
        """
        if not isinstance(self.client, openai.AsyncOpenAI):
            raise TypeError("Async operations require an AsyncOpenAI client")

        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        # Create streaming completion with retries
        async for attempt in AsyncRetrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create streaming completion
                    stream = await self.client.chat.completions.create(**request_params)

                    # Process stream chunks
                    async for chunk in stream:
                        # Run post-chunk hooks
                        if hooks:
                            hooks.post_completion(chunk)

                        # Process chunk
                        result = self.process_response(
                            chunk,
                            response_model,
                            mode,
                            validation_context=validation_context,
                            context=context,
                            strict=strict,
                            partial=True,
                        )

                        if result is not None:
                            yield result

                except Exception as e:
                    self.handle_error(e, None)
                    raise

    def create_iterable(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        retry_config: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> Generator[T, None, None]:
        """Create a streaming completion that yields list items.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            retry_config: Number of retries or a Retrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            Generator yielding list items
        """
        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        # Create streaming completion with retries
        for attempt in Retrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create streaming completion
                    stream = self.client.chat.completions.create(**request_params)

                    # Process stream chunks
                    for chunk in stream:
                        # Run post-chunk hooks
                        if hooks:
                            hooks.post_completion(chunk)

                        # Process chunk
                        result = self.process_response(
                            chunk,
                            response_model,
                            mode,
                            validation_context=validation_context,
                            context=context,
                            strict=strict,
                            iterable=True,
                        )

                        if result is not None:
                            yield result

                except Exception as e:
                    self.handle_error(e, None)
                    raise

    async def async_create_iterable(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        retry_config: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        """Create a streaming completion that yields list items asynchronously.

        Args:
            messages: List of chat messages
            response_model: Expected response model type
            retry_config: Number of retries or an AsyncRetrying instance
            validation_context: Context for model validation
            context: Additional context for processing
            strict: Whether to use strict validation
            hooks: Hooks for the completion process
            mode: Request mode (default: Mode.FUNCTIONS)
            **kwargs: Additional provider-specific options

        Returns:
            AsyncGenerator yielding list items
        """
        if not isinstance(self.client, openai.AsyncOpenAI):
            raise TypeError("Async operations require an AsyncOpenAI client")

        # Prepare request parameters
        request_params = self.prepare_request(response_model, mode, **kwargs)
        request_params["messages"] = messages
        request_params["stream"] = True

        # Create streaming completion with retries
        async for attempt in AsyncRetrying(retry_config):
            with attempt:
                try:
                    # Run pre-completion hooks
                    if hooks:
                        hooks.pre_completion(request_params)

                    # Create streaming completion
                    stream = await self.client.chat.completions.create(**request_params)

                    # Process stream chunks
                    async for chunk in stream:
                        # Run post-chunk hooks
                        if hooks:
                            hooks.post_completion(chunk)

                        # Process chunk
                        result = self.process_response(
                            chunk,
                            response_model,
                            mode,
                            validation_context=validation_context,
                            context=context,
                            strict=strict,
                            iterable=True,
                        )

                        if result is not None:
                            yield result

                except Exception as e:
                    self.handle_error(e, None)
                    raise
