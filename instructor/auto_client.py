from __future__ import annotations
from typing import Any, Union, Literal, overload, Optional
from instructor.client import AsyncInstructor, Instructor, CachedInstructor, CachedAsyncInstructor
import instructor
from instructor.models import KnownModelName

# Type alias for the return type
InstructorType = Union[Instructor, AsyncInstructor, CachedInstructor, CachedAsyncInstructor]


# List of supported providers
supported_providers = [
    "openai",
    "azure_openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "perplexity",
    "groq",
    "writer",
    "bedrock",
    "cerebras",
    "fireworks",
    "vertexai",
    "generative-ai",
    "ollama",
]


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[True] = True,
    mode: Union[instructor.Mode, None] = None,
    cache: Optional[Any] = None,
    **kwargs: Any,
) -> AsyncInstructor | CachedAsyncInstructor: ...


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[False] = False,
    mode: Union[instructor.Mode, None] = None,
    cache: Optional[Any] = None,
    **kwargs: Any,
) -> Instructor | CachedInstructor: ...


@overload
def from_provider(
    model: str, 
    async_client: Literal[True] = True, 
    mode: Union[instructor.Mode, None] = None,
    cache: Optional[Any] = None,
    **kwargs: Any
) -> AsyncInstructor | CachedAsyncInstructor: ...


@overload
def from_provider(
    model: str, 
    async_client: Literal[False] = False, 
    mode: Union[instructor.Mode, None] = None,
    cache: Optional[Any] = None,
    **kwargs: Any
) -> Instructor | CachedInstructor: ...


def from_provider(
    model: Union[str, KnownModelName],  # noqa: UP007
    async_client: bool = False,
    mode: Union[instructor.Mode, None] = None,  # noqa: ARG001, UP007
    cache: Optional[Any] = None,
    **kwargs: Any,
) -> Union[Instructor, AsyncInstructor, CachedInstructor, CachedAsyncInstructor]:  # noqa: UP007
    """Create an Instructor client from a model string.

    Args:
        model: String in format "provider/model-name"
              (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro")
        async_client: Whether to return an async client
        mode: Override the default mode for the provider. If not specified, uses the
              recommended default mode for each provider.
        cache: Optional cache backend for response caching. Can be LRUCache, 
               RedisCache, DiskCache, or any object implementing the CacheBackend interface.
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        Instructor, AsyncInstructor, CachedInstructor, or CachedAsyncInstructor instance

    Raises:
        ValueError: If provider is not supported or model string is invalid
        ImportError: If required package for provider is not installed

    Examples:
        >>> import instructor
        >>> from instructor.cache import LRUCache, RedisCache, DiskCache
        >>> 
        >>> # Basic usage without cache
        >>> client = instructor.from_provider("openai/gpt-4")
        >>> 
        >>> # With LRU cache
        >>> cache = LRUCache(maxsize=1000)
        >>> client = instructor.from_provider("openai/gpt-4", cache=cache)
        >>> 
        >>> # With Redis cache
        >>> cache = RedisCache(redis_url="redis://localhost", ttl=3600)
        >>> client = instructor.from_provider("openai/gpt-4", cache=cache)
        >>> 
        >>> # With disk cache
        >>> cache = DiskCache(cache_dir="./cache", ttl=3600)
        >>> client = instructor.from_provider("openai/gpt-4", cache=cache)
        >>> 
        >>> # Async clients with cache
        >>> async_client = instructor.from_provider("openai/gpt-4", async_client=True, cache=cache)
    """
    
    def _create_cached_client(base_client, cache_backend, is_async=False):
        """Helper function to create cached client from base client."""
        if is_async:
            return CachedAsyncInstructor(
                cache=cache_backend,
                client=base_client.client,
                create=base_client.create_fn,
                mode=base_client.mode,
                provider=base_client.provider,
                hooks=base_client.hooks,
                **kwargs
            )
        else:
            return CachedInstructor(
                cache=cache_backend,
                client=base_client.client,
                create=base_client.create_fn,
                mode=base_client.mode,
                provider=base_client.provider,
                hooks=base_client.hooks,
                **kwargs
            )
    
    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            'Model string must be in format "provider/model-name" '
            '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
        ) from None

    # Create base client first, then wrap with cache if provided
    base_client = None

    if provider == "openai":
        try:
            import openai
            from instructor import from_openai

            client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
            base_client = from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
        except ImportError:
            from instructor.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the OpenAI provider. "
                "Install it with `pip install openai`."
            ) from None

    elif provider == "azure_openai":
        try:
            import os
            from openai import AzureOpenAI, AsyncAzureOpenAI
            from instructor import from_openai

            # Get required Azure OpenAI configuration from environment
            api_key = kwargs.pop("api_key", os.environ.get("AZURE_OPENAI_API_KEY"))
            azure_endpoint = kwargs.pop("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT"))
            api_version = kwargs.pop("api_version", "2024-02-01")

            if not api_key:
                from instructor.exceptions import ConfigurationError
                raise ConfigurationError(
                    "AZURE_OPENAI_API_KEY is not set. "
                    "Set it with `export AZURE_OPENAI_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
                )

            if not azure_endpoint:
                from instructor.exceptions import ConfigurationError
                raise ConfigurationError(
                    "AZURE_OPENAI_ENDPOINT is not set. "
                    "Set it with `export AZURE_OPENAI_ENDPOINT=<your-endpoint>` or pass it as kwarg azure_endpoint=<your-endpoint>"
                )

            client = (
                AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                )
                if async_client
                else AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                )
            )
            base_client = from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
        except ImportError:
            from instructor.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the Azure OpenAI provider. "
                "Install it with `pip install openai`."
            ) from None

    elif provider == "anthropic":
        try:
            import anthropic
            from instructor import from_anthropic

            client = (
                anthropic.AsyncAnthropic() if async_client else anthropic.Anthropic()
            )
            max_tokens = kwargs.pop("max_tokens", 4096)
            base_client = from_anthropic(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.ANTHROPIC_TOOLS,
                max_tokens=max_tokens,
                **kwargs,
            )
        except ImportError:
            import_err = ImportError(
                "The anthropic package is required to use the Anthropic provider. "
                "Install it with `pip install anthropic`."
            )
            raise import_err from None

    elif provider == "google":
        try:
            import google.genai as genai  # type: ignore
            from instructor import from_genai

            client = genai.Client(
                vertexai=False
                if kwargs.get("vertexai") is None
                else kwargs.get("vertexai"),
                **kwargs,
            )  # type: ignore
            if async_client:
                base_client = from_genai(client, use_async=True, model=model_name, **kwargs)  # type: ignore
            else:
                base_client = from_genai(client, model=model_name, **kwargs)  # type: ignore
        except ImportError:
            import_err = ImportError(
                "The google-genai package is required to use the Google provider. "
                "Install it with `pip install google-genai`."
            )
            raise import_err from None

    elif provider == "mistral":
        try:
            from mistralai import Mistral  # type: ignore
            from instructor import from_mistral
            import os

            if os.environ.get("MISTRAL_API_KEY"):
                client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )

            if async_client:
                base_client = from_mistral(client, model=model_name, use_async=True, **kwargs)
            else:
                base_client = from_mistral(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The mistralai package is required to use the Mistral provider. "
                "Install with: pip install mistralai`."
            )
            raise import_err from None

    elif provider == "cohere":
        try:
            import cohere
            from instructor import from_cohere

            client = cohere.AsyncClient() if async_client else cohere.Client()
            base_client = from_cohere(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The cohere package is required to use the Cohere provider. "
                "Install it with `pip install cohere`."
            )
            raise import_err from None

    elif provider == "perplexity":
        try:
            import openai
            from instructor import from_perplexity
            import os

            if os.environ.get("PERPLEXITY_API_KEY"):
                api_key = os.environ.get("PERPLEXITY_API_KEY")
            elif kwargs.get("api_key"):
                api_key = kwargs.get("api_key")
            else:
                raise ValueError(
                    "PERPLEXITY_API_KEY is not set. "
                    "Set it with `export PERPLEXITY_API_KEY=<your-api-key>` or pass it as a kwarg api_key=<your-api-key>"
                )

            client = (
                openai.AsyncOpenAI(
                    api_key=api_key, base_url="https://api.perplexity.ai"
                )
                if async_client
                else openai.OpenAI(
                    api_key=api_key, base_url="https://api.perplexity.ai"
                )
            )
            base_client = from_perplexity(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The openai package is required to use the Perplexity provider. "
                "Install it with `pip install openai`."
            )
            raise import_err from None

    elif provider == "groq":
        try:
            import groq
            from instructor import from_groq

            client = groq.AsyncGroq() if async_client else groq.Groq()
            base_client = from_groq(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The groq package is required to use the Groq provider. "
                "Install it with `pip install groq`."
            )
            raise import_err from None

    elif provider == "writer":
        try:
            from writerai import AsyncWriter, Writer
            from instructor import from_writer

            client = AsyncWriter() if async_client else Writer()
            base_client = from_writer(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The writerai package is required to use the Writer provider. "
                "Install it with `pip install writer-sdk`."
            )
            raise import_err from None

    elif provider == "bedrock":
        try:
            import boto3
            from instructor import from_bedrock

            client = boto3.client("bedrock-runtime")
            base_client = from_bedrock(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The boto3 package is required to use the AWS Bedrock provider. "
                "Install it with `pip install boto3`."
            )
            raise import_err from None

    elif provider == "cerebras":
        try:
            from cerebras.cloud.sdk import AsyncCerebras, Cerebras
            from instructor import from_cerebras

            client = AsyncCerebras() if async_client else Cerebras()
            base_client = from_cerebras(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The cerebras package is required to use the Cerebras provider. "
                "Install it with `pip install cerebras`."
            )
            raise import_err from None

    elif provider == "fireworks":
        try:
            from fireworks.client import AsyncFireworks, Fireworks
            from instructor import from_fireworks

            client = AsyncFireworks() if async_client else Fireworks()
            base_client = from_fireworks(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The fireworks-ai package is required to use the Fireworks provider. "
                "Install it with `pip install fireworks-ai`."
            )
            raise import_err from None

    elif provider == "vertexai":
        try:
            import vertexai.generative_models as gm
            from instructor import from_vertexai

            client = gm.GenerativeModel(model_name=model_name)
            base_client = from_vertexai(client, use_async=async_client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The google-cloud-aiplatform package is required to use the VertexAI provider. "
                "Install it with `pip install google-cloud-aiplatform`."
            )
            raise import_err from None

    elif provider == "generative-ai":
        try:
            from google.generativeai import GenerativeModel
            from instructor import from_gemini

            client = GenerativeModel(model_name=model_name)
            if async_client:
                base_client = from_gemini(client, use_async=True, **kwargs)  # type: ignore
            else:
                base_client = from_gemini(client, **kwargs)  # type: ignore
        except ImportError:
            import_err = ImportError(
                "The google-generativeai package is required to use the Google GenAI provider. "
                "Install it with `pip install google-genai`."
            )
            raise import_err from None

    elif provider == "ollama":
        try:
            import openai
            from instructor import from_openai

            # Get base_url from kwargs or use default
            base_url = kwargs.pop("base_url", "http://localhost:11434/v1")
            api_key = kwargs.pop("api_key", "ollama")  # required but unused

            client = (
                openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
                if async_client
                else openai.OpenAI(base_url=base_url, api_key=api_key)
            )

            # Models that support function calling (tools mode)
            tool_capable_models = {
                "llama3.1",
                "llama3.2",
                "llama4",
                "mistral-nemo",
                "firefunction-v2",
                "command-r-plus",
                "qwen2.5",
                "qwen2.5-coder",
                "qwen3",
                "devstral",
            }

            # Check if model supports tools by looking at model name
            supports_tools = any(
                capable_model in model_name.lower()
                for capable_model in tool_capable_models
            )

            default_mode = (
                instructor.Mode.TOOLS if supports_tools else instructor.Mode.JSON
            )

            base_client = from_openai(
                client,
                model=model_name,
                mode=mode if mode else default_mode,
                **kwargs,
            )
        except ImportError:
            from instructor.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the Ollama provider. "
                "Install it with `pip install openai`."
            ) from None

    else:
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )

    # If cache is provided, wrap the base client with caching
    if cache is not None:
        # Validate cache backend
        try:
            from instructor.cache.base import CacheBackend
            if not hasattr(cache, 'get') or not hasattr(cache, 'set'):
                raise ValueError(
                    f"Cache backend must implement get() and set() methods. "
                    f"Got {type(cache)}. Use LRUCache, RedisCache, DiskCache, or implement CacheBackend interface."
                )
        except ImportError:
            # If CacheBackend not available, do basic duck-typing check
            if not hasattr(cache, 'get') or not hasattr(cache, 'set'):
                raise ValueError(
                    f"Cache backend must implement get() and set() methods. "
                    f"Got {type(cache)}."
                )
        
        return _create_cached_client(base_client, cache, async_client)
    
    return base_client
