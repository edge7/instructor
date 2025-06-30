from __future__ import annotations

from functools import partial
from typing import Any, Callable, TYPE_CHECKING
from .base import BaseProvider

if TYPE_CHECKING:
    import instructor
    from instructor.utils import Provider, get_provider
    from instructor.hooks import Hooks
    try:
        import openai
    except ImportError:
        openai = None  # type: ignore


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI clients."""
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def validate_client(self, client: Any) -> bool:
        """Validate that the client is an OpenAI client."""
        try:
            import openai
            return isinstance(client, (openai.OpenAI, openai.AsyncOpenAI))
        except ImportError:
            return False
    
    def get_instructor_class(self, client: Any) -> type[Any]:  # type[instructor.Instructor] | type[instructor.AsyncInstructor]
        """Get the appropriate Instructor class based on client type."""
        try:
            import openai
            import instructor
            if isinstance(client, openai.AsyncOpenAI):
                return instructor.AsyncInstructor
            elif isinstance(client, openai.OpenAI):
                return instructor.Instructor
            else:
                raise ValueError("Client must be an instance of openai.OpenAI or openai.AsyncOpenAI")
        except ImportError:
            raise ImportError("OpenAI and instructor packages are required for OpenAI provider")
    
    def get_provider_enum(self) -> Any:  # Provider
        """Get the Provider enum value for OpenAI."""
        from instructor.utils import Provider
        return Provider.OPENAI
    
    def get_create_function(self, client: Any, mode: Any) -> Callable[..., Any]:  # mode: instructor.Mode
        """Get the create function for OpenAI clients."""
        try:
            import openai
            import instructor
            if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
                raise ValueError("Client must be an instance of openai.OpenAI or openai.AsyncOpenAI")
            
            # Handle special response modes
            if mode in {
                instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                instructor.Mode.RESPONSES_TOOLS,
            }:
                if isinstance(client, openai.OpenAI):
                    return instructor.patch(
                        create=partial(self._map_chat_completion_to_response, client=client),
                        mode=mode,
                    )
                else:  # AsyncOpenAI
                    return instructor.patch(
                        create=partial(self._async_map_chat_completion_to_response, client=client),
                        mode=mode,
                    )
            else:
                # Standard modes
                return instructor.patch(
                    create=client.chat.completions.create,
                    mode=mode,
                )
        except ImportError:
            raise ImportError("OpenAI and instructor packages are required for OpenAI provider")
    
    def validate_mode(self, mode: Any) -> bool:  # mode: instructor.Mode
        """Validate that the mode is supported by OpenAI."""
        import instructor
        # Get the provider type from the client
        supported_modes = {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.FUNCTIONS,
            instructor.Mode.PARALLEL_TOOLS,
            instructor.Mode.MD_JSON,
            instructor.Mode.TOOLS_STRICT,
            instructor.Mode.JSON_O1,
            instructor.Mode.RESPONSES_TOOLS,
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }
        return mode in supported_modes
    
    def _map_chat_completion_to_response(self, messages, client, *args, **kwargs) -> Any:
        """Map chat completion to response for sync clients."""
        return client.responses.create(
            *args,
            input=messages,
            **kwargs,
        )
    
    async def _async_map_chat_completion_to_response(self, messages, client, *args, **kwargs) -> Any:
        """Map chat completion to response for async clients."""
        return await client.responses.create(
            *args,
            input=messages,
            **kwargs,
        )
    
    def create_instructor(
        self,
        client: Any,
        mode: Any = None,  # instructor.Mode = instructor.Mode.TOOLS,
        hooks: Any | None = None,  # Hooks | None = None,
        **kwargs: Any,
    ) -> Any:  # instructor.Instructor | instructor.AsyncInstructor
        """Create an Instructor instance for OpenAI clients with additional validation."""
        try:
            import openai
            import instructor
            from instructor.utils import Provider, get_provider
            
            # Set default mode if not provided
            if mode is None:
                mode = instructor.Mode.TOOLS
            
            if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
                import warnings
                warnings.warn(
                    "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. "
                    "Unexpected behavior may occur with other client types.",
                    stacklevel=2,
                )
            
            # Determine provider type from client base_url if available
            provider_enum = self.get_provider_enum()
            if hasattr(client, "base_url"):
                provider_enum = get_provider(str(client.base_url))
                
                # Validate mode for specific provider types
                if provider_enum == Provider.OPENROUTER:
                    if mode not in {
                        instructor.Mode.TOOLS,
                        instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS,
                        instructor.Mode.JSON,
                    }:
                        raise ValueError(f"Mode {mode} is not supported for OpenRouter")
                
                elif provider_enum in {Provider.ANYSCALE, Provider.TOGETHER}:
                    if mode not in {
                        instructor.Mode.TOOLS,
                        instructor.Mode.JSON,
                        instructor.Mode.JSON_SCHEMA,
                        instructor.Mode.MD_JSON,
                    }:
                        raise ValueError(f"Mode {mode} is not supported for {provider_enum.value}")
                
                elif provider_enum in {Provider.OPENAI, Provider.DATABRICKS}:
                    if not self.validate_mode(mode):
                        raise ValueError(f"Mode {mode} is not supported for {provider_enum.value}")
            
            instructor_class = self.get_instructor_class(client)
            create_function = self.get_create_function(client, mode)
            
            return instructor_class(
                client=client,
                create=create_function,
                mode=mode,
                provider=provider_enum,
                hooks=hooks,
                **kwargs,
            )
            
        except ImportError:
            raise ImportError("OpenAI and instructor packages are required for OpenAI provider")


# Create and register the OpenAI provider
openai_provider = OpenAIProvider()

# Register the provider when this module is imported
try:
    from .base import registry
    registry.register(openai_provider)
except ImportError:
    # Base registry not available
    pass

def from_openai(
    client,
    mode: Any = None,  # instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Any:  # instructor.Instructor | instructor.AsyncInstructor
    """
    Create an Instructor instance from an OpenAI client.
    
    This function maintains backward compatibility with the existing API
    while using the new provider system internally.
    
    Args:
        client: OpenAI client (sync or async)
        mode: Instructor mode to use
        **kwargs: Additional keyword arguments
        
    Returns:
        Instructor or AsyncInstructor instance
    """
    return openai_provider.create_instructor(client, mode, **kwargs)