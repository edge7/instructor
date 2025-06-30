# Provider Abstraction System

This directory contains the new provider abstraction system for the instructor library. The provider system allows for clean separation of different LLM provider implementations while maintaining backward compatibility with existing APIs.

## Architecture

### Base Provider (`base.py`)

The `BaseProvider` abstract class defines the interface that all provider implementations must follow:

- `provider_name`: Unique identifier for the provider
- `validate_client()`: Check if a client is compatible with this provider
- `get_create_function()`: Get the appropriate create function for API calls
- `get_instructor_class()`: Get the Instructor or AsyncInstructor class to use
- `get_provider_enum()`: Get the Provider enum value
- `validate_mode()`: Check if a mode is supported by this provider
- `create_instructor()`: Create an Instructor instance (with default implementation)

### Provider Registry

The `ProviderRegistry` class manages provider implementations:

- `register()`: Register a new provider
- `get_provider()`: Get provider by name
- `get_provider_for_client()`: Find provider for a specific client
- `list_providers()`: List all registered providers

### OpenAI Provider (`client_openai.py`)

The `OpenAIProvider` implements the base provider interface for OpenAI clients:

- Supports both sync (`openai.OpenAI`) and async (`openai.AsyncOpenAI`) clients
- Handles all OpenAI-compatible modes (TOOLS, JSON, FUNCTIONS, etc.)
- Supports OpenAI-compatible providers (OpenRouter, Anyscale, Together, etc.)
- Maintains full backward compatibility with existing `from_openai()` function

## Usage

### Backward Compatibility

The existing API continues to work unchanged:

```python
import instructor
import openai

client = openai.OpenAI()
instructor_client = instructor.from_openai(client)
```

### New Provider API

You can also use the provider system directly:

```python
from instructor.providers.client_openai import openai_provider
import openai

client = openai.OpenAI()
instructor_client = openai_provider.create_instructor(client)
```

### Custom Provider

Create your own provider by implementing the `BaseProvider` interface:

```python
from instructor.providers.base import BaseProvider, registry
import instructor

class MyProvider(BaseProvider):
    @property
    def provider_name(self) -> str:
        return "my_provider"
    
    def validate_client(self, client) -> bool:
        return isinstance(client, MyClientType)
    
    def get_create_function(self, client, mode):
        return instructor.patch(create=client.complete, mode=mode)
    
    def get_instructor_class(self, client):
        return instructor.Instructor
    
    def get_provider_enum(self):
        from instructor.utils import Provider
        return Provider.UNKNOWN
    
    def validate_mode(self, mode) -> bool:
        return mode in {instructor.Mode.JSON, instructor.Mode.TOOLS}

# Register your provider
my_provider = MyProvider()
registry.register(my_provider)

# Use it
client = MyClientType()
instructor_client = my_provider.create_instructor(client)
```

## Design Principles

1. **Backward Compatibility**: Existing code should continue to work without changes
2. **Conditional Imports**: Providers only import their dependencies when needed
3. **Clean Separation**: Each provider is self-contained and testable
4. **Extensibility**: Easy to add new providers without modifying core code
5. **Type Safety**: Full type hints with proper TYPE_CHECKING guards

## Implementation Notes

### Import Strategy

The provider system uses careful import management to avoid loading heavy dependencies:

- Core instructor types are imported only in `TYPE_CHECKING` blocks
- Provider-specific dependencies (like `openai`) are imported only when needed
- The base provider system can be imported without any LLM dependencies

### Error Handling

Each provider gracefully handles missing dependencies:

- `validate_client()` returns `False` if dependencies are missing
- Other methods raise descriptive `ImportError` messages
- The system degrades gracefully when providers are unavailable

### Migration Strategy

The migration maintains 100% backward compatibility:

1. Existing `from_openai()` function tries the new provider system first
2. Falls back to original implementation if provider system fails
3. No changes required in downstream code
4. New provider features can be adopted incrementally

## Testing

The provider system includes comprehensive testing:

- Unit tests for base provider functionality
- Integration tests for OpenAI provider
- Backward compatibility verification
- Error handling validation

Run tests with:

```bash
python test_provider_migration.py
```

## Future Extensions

This architecture enables future enhancements:

- Dynamic provider discovery
- Provider-specific configuration
- Performance optimizations per provider
- Advanced provider selection logic
- Plugin-based provider loading