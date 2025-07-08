"""OpenAI provider implementation."""

from typing import Any, Optional, TypeVar, Union
from collections.abc import Generator, AsyncGenerator
from collections.abc import Iterable
from pydantic import BaseModel
import openai
from openai.types.chat import ChatCompletion
from tenacity import AsyncRetrying, Retrying

from ..base import BaseProvider
from ...mode import Mode
from ...dsl.partial import Partial
from ...hooks import Hooks
from .response import process_openai_response

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

    def _check_async_client(self) -> None:
        """Check if client is AsyncOpenAI."""
        if not isinstance(self.client, openai.AsyncOpenAI):
            raise TypeError("Async operations require an AsyncOpenAI client")

    def _prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare kwargs for streaming operations."""
        kwargs = kwargs.copy()
        kwargs["stream"] = True
        return kwargs

    async def async_create_with_completion(
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
    ) -> tuple[T, ChatCompletion]:
        """Create a completion asynchronously and return both the processed response and raw completion."""
        self._check_async_client()
        
        request = self.prepare_request(
            response_model=response_model,
            mode=mode,
            messages=messages,
            **kwargs,
        )
        
        response = await self._retry_async(
            lambda: self.client.chat.completions.create(**request),
            max_retries=max_retries,
        )
        
        result = self.process_response(
            response=response,
            response_model=response_model,
            mode=mode,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=hooks,
        )
        
        return result, response

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
        kwargs = self._prepare_streaming(kwargs)
        request = self.prepare_request(
            response_model=response_model,
            mode=mode,
            messages=messages,
            **kwargs,
        )
        
        stream = self._retry_sync(
            lambda: self.client.chat.completions.create(**request),
            max_retries=max_retries,
        )
        
        for response in stream:
            result = self.process_response(
                response=response,
                response_model=response_model,
                mode=mode,
                validation_context=validation_context,
                context=context,
                strict=strict,
                hooks=hooks,
            )
            if result is not None:
                yield result

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
        self._check_async_client()
        kwargs = self._prepare_streaming(kwargs)
        
        request = self.prepare_request(
            response_model=response_model,
            mode=mode,
            messages=messages,
            **kwargs,
        )
        
        stream = await self._retry_async(
            lambda: self.client.chat.completions.create(**request),
            max_retries=max_retries,
        )
        
        async for response in stream:
            result = self.process_response(
                response=response,
                response_model=response_model,
                mode=mode,
                validation_context=validation_context,
                context=context,
                strict=strict,
                hooks=hooks,
            )
            if result is not None:
                yield result

    def create_iterable(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> Generator[T, None, None]:
        """Create a streaming completion that yields list items."""
        kwargs["stream"] = True
        return super().create_iterable(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=hooks,
            mode=mode,
            **kwargs,
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
        mode: Mode = Mode.FUNCTIONS,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        """Create a streaming completion that yields list items asynchronously."""
        if not isinstance(self.client, openai.AsyncOpenAI):
            raise TypeError("Async operations require an AsyncOpenAI client")
        kwargs["stream"] = True
        async for item in super().async_create_iterable(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=hooks,
            mode=mode,
            **kwargs,
        ):
            yield item
