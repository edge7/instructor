from __future__ import annotations

from typing import Any, Callable, Protocol

from pydantic import BaseModel

from .mode import Mode
from .process_response import process_response, process_response_async


class ProviderAdapter(Protocol):
    """Interface for provider-specific adapters."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
        mode: Mode,
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]: ...

    def reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
        mode: Mode,
    ) -> dict[str, Any]: ...

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | None,
        context: dict[str, Any] | None,
        strict: bool | None,
        mode: Mode,
        *,
        stream: bool,
    ) -> BaseModel | Any: ...

    async def parse_response_async(
        self,
        response: Any,
        response_model: type[BaseModel] | None,
        context: dict[str, Any] | None,
        strict: bool | None,
        mode: Mode,
        *,
        stream: bool,
    ) -> BaseModel | Any: ...


class RegistryProviderAdapter:
    """Adapter implementation backed by a handler registry."""

    def __init__(self, handlers: dict[Mode, dict[str, Callable]]):
        self.handlers = handlers

    def _get(self, mode: Mode, key: str) -> Callable:
        try:
            return self.handlers[mode][key]
        except KeyError as e:
            raise ValueError(f"Unsupported mode {mode}") from e

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
        mode: Mode,
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        handler = self._get(mode, "response")
        return handler(response_model, kwargs)

    def reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
        mode: Mode,
    ) -> dict[str, Any]:
        handler = self._get(mode, "reask")
        return handler(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | None,
        context: dict[str, Any] | None,
        strict: bool | None,
        mode: Mode,
        *,
        stream: bool,
    ) -> BaseModel | Any:
        return process_response(
            response=response,
            response_model=response_model,
            validation_context=context,
            strict=strict,
            mode=mode,
            stream=stream,
        )

    async def parse_response_async(
        self,
        response: Any,
        response_model: type[BaseModel] | None,
        context: dict[str, Any] | None,
        strict: bool | None,
        mode: Mode,
        *,
        stream: bool,
    ) -> BaseModel | Any:
        return await process_response_async(
            response=response,
            response_model=response_model,
            validation_context=context,
            strict=strict,
            mode=mode,
            stream=stream,
        )


from .utils.anthropic import ANTHROPIC_HANDLERS
from .utils.openai import OPENAI_HANDLERS
from .utils.google import GOOGLE_HANDLERS
from .utils.cohere import COHERE_HANDLERS
from .utils.mistral import MISTRAL_HANDLERS
from .utils.bedrock import BEDROCK_HANDLERS
from .utils.fireworks import FIREWORKS_HANDLERS
from .utils.cerebras import CEREBRAS_HANDLERS
from .utils.writer import WRITER_HANDLERS
from .utils.perplexity import PERPLEXITY_HANDLERS
from .utils.providers import Provider

OPENAI_ADAPTER = RegistryProviderAdapter(OPENAI_HANDLERS)
ANTHROPIC_ADAPTER = RegistryProviderAdapter(ANTHROPIC_HANDLERS)
GOOGLE_ADAPTER = RegistryProviderAdapter(GOOGLE_HANDLERS)
COHERE_ADAPTER = RegistryProviderAdapter(COHERE_HANDLERS)
MISTRAL_ADAPTER = RegistryProviderAdapter(MISTRAL_HANDLERS)
BEDROCK_ADAPTER = RegistryProviderAdapter(BEDROCK_HANDLERS)
FIREWORKS_ADAPTER = RegistryProviderAdapter(FIREWORKS_HANDLERS)
CEREBRAS_ADAPTER = RegistryProviderAdapter(CEREBRAS_HANDLERS)
WRITER_ADAPTER = RegistryProviderAdapter(WRITER_HANDLERS)
PERPLEXITY_ADAPTER = RegistryProviderAdapter(PERPLEXITY_HANDLERS)

ADAPTER_REGISTRY: dict[Provider, RegistryProviderAdapter] = {
    Provider.OPENAI: OPENAI_ADAPTER,
    Provider.ANTHROPIC: ANTHROPIC_ADAPTER,
    Provider.GEMINI: GOOGLE_ADAPTER,
    Provider.GENAI: GOOGLE_ADAPTER,
    Provider.VERTEXAI: GOOGLE_ADAPTER,
    Provider.COHERE: COHERE_ADAPTER,
    Provider.MISTRAL: MISTRAL_ADAPTER,
    Provider.BEDROCK: BEDROCK_ADAPTER,
    Provider.FIREWORKS: FIREWORKS_ADAPTER,
    Provider.CEREBRAS: CEREBRAS_ADAPTER,
    Provider.WRITER: WRITER_ADAPTER,
    Provider.PERPLEXITY: PERPLEXITY_ADAPTER,
    Provider.GROQ: OPENAI_ADAPTER,
    Provider.OPENROUTER: OPENAI_ADAPTER,
    Provider.XAI: OPENAI_ADAPTER,
}
