# Response Handlers Refactor Proposal

## Current State Analysis

The current response handler system in `instructor/process_response.py` has grown organically and exhibits several architectural issues:

### Problems Identified

1. **Massive File Size**: The `process_response.py` file is 1,192 lines and contains 25+ handler functions
2. **Code Duplication**: Many handlers share similar patterns but duplicate implementation
3. **Poor Organization**: All handlers are in a single file with no logical grouping
4. **Inconsistent Interfaces**: Different handlers have slightly different signatures and behaviors
5. **Lack of Extensibility**: Adding new providers requires modifying the core dispatch logic
6. **Mixed Responsibilities**: Response model preparation, kwargs transformation, and provider-specific logic are intermingled
7. **Hard to Test**: Individual handlers are difficult to unit test in isolation
8. **No Abstraction**: Similar provider types (e.g., tool-based vs JSON-based) share no common abstractions

### Current Architecture

```python
# Current monolithic approach
mode_handlers = {
    Mode.TOOLS: handle_tools,
    Mode.JSON: handle_json_modes,
    Mode.ANTHROPIC_TOOLS: handle_anthropic_tools,
    # ... 25+ more handlers
}

response_model, new_kwargs = mode_handlers[mode](response_model, new_kwargs)
```

## Proposed Refactor

### 1. Extract Handler Classes with Strategy Pattern

Create a hierarchy of handler classes that implement a common interface:

```python
# instructor/handlers/base.py
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Tuple, Dict

T = TypeVar("T")

class BaseResponseHandler(ABC):
    """Base class for all response handlers."""
    
    @abstractmethod
    def handle(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        """Process response model and kwargs for the specific provider/mode."""
        pass
    
    @property
    @abstractmethod
    def supported_modes(self) -> set[Mode]:
        """Return the modes this handler supports."""
        pass
    
    def validate_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate kwargs for this handler. Override if needed."""
        pass


class ToolBasedHandler(BaseResponseHandler):
    """Base class for tool-based response handlers."""
    
    def handle(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        self.validate_kwargs(kwargs)
        kwargs = self._prepare_tools(response_model, kwargs)
        kwargs = self._set_tool_choice(response_model, kwargs)
        return response_model, kwargs
    
    @abstractmethod
    def _prepare_tools(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare tools configuration for the provider."""
        pass
    
    @abstractmethod
    def _set_tool_choice(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Set tool choice configuration for the provider."""
        pass


class JsonBasedHandler(BaseResponseHandler):
    """Base class for JSON-based response handlers."""
    
    def handle(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        self.validate_kwargs(kwargs)
        kwargs = self._prepare_json_schema(response_model, kwargs)
        kwargs = self._inject_system_message(response_model, kwargs)
        return response_model, kwargs
    
    @abstractmethod
    def _prepare_json_schema(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare JSON schema configuration."""
        pass
    
    def _inject_system_message(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add JSON schema instructions to system message."""
        # Common implementation for most JSON handlers
        pass
```

### 2. Organize Handlers by Provider

```
instructor/handlers/
├── __init__.py
├── base.py
├── openai/
│   ├── __init__.py
│   ├── tools.py      # OpenAI tool handlers
│   ├── json.py       # OpenAI JSON handlers
│   └── functions.py  # Deprecated functions handler
├── anthropic/
│   ├── __init__.py
│   ├── tools.py
│   └── json.py
├── vertexai/
│   ├── __init__.py
│   ├── tools.py
│   ├── json.py
│   └── parallel.py
├── cohere/
│   ├── __init__.py
│   └── handlers.py
├── bedrock/
│   ├── __init__.py
│   ├── tools.py
│   └── json.py
└── registry.py       # Handler registration and discovery
```

### 3. Implement Provider-Specific Handlers

```python
# instructor/handlers/openai/tools.py
from ..base import ToolBasedHandler
from ...mode import Mode

class OpenAIToolsHandler(ToolBasedHandler):
    """Handler for OpenAI tools mode."""
    
    @property
    def supported_modes(self) -> set[Mode]:
        return {Mode.TOOLS}
    
    def _prepare_tools(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["tools"] = [{
            "type": "function",
            "function": response_model.openai_schema,
        }]
        return kwargs
    
    def _set_tool_choice(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": response_model.openai_schema["name"]},
        }
        return kwargs


class OpenAIToolsStrictHandler(OpenAIToolsHandler):
    """Handler for OpenAI strict tools mode."""
    
    @property
    def supported_modes(self) -> set[Mode]:
        return {Mode.TOOLS_STRICT}
    
    def _prepare_tools(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        from openai import pydantic_function_tool
        
        response_model_schema = pydantic_function_tool(response_model)
        response_model_schema["function"]["strict"] = True
        kwargs["tools"] = [response_model_schema]
        return kwargs


# instructor/handlers/anthropic/tools.py
class AnthropicToolsHandler(ToolBasedHandler):
    """Handler for Anthropic tools mode."""
    
    @property
    def supported_modes(self) -> set[Mode]:
        return {Mode.ANTHROPIC_TOOLS}
    
    def _prepare_tools(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["tools"] = [response_model.anthropic_schema]
        return kwargs
    
    def _set_tool_choice(self, response_model: type[T], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["tool_choice"] = {
            "type": "tool",
            "name": response_model.__name__,
        }
        return kwargs
    
    def handle(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        # Anthropic-specific message handling
        from ...utils import extract_system_messages, combine_system_messages
        
        response_model, kwargs = super().handle(response_model, kwargs)
        
        system_messages = extract_system_messages(kwargs.get("messages", []))
        if system_messages:
            kwargs["system"] = combine_system_messages(
                kwargs.get("system"), system_messages
            )
        
        kwargs["messages"] = [
            m for m in kwargs.get("messages", []) if m["role"] != "system"
        ]
        
        return response_model, kwargs
```

### 4. Create Handler Registry

```python
# instructor/handlers/registry.py
from typing import Dict, Type
from .base import BaseResponseHandler
from ..mode import Mode

class HandlerRegistry:
    """Registry for response handlers."""
    
    def __init__(self):
        self._handlers: Dict[Mode, BaseResponseHandler] = {}
    
    def register(self, handler: BaseResponseHandler) -> None:
        """Register a handler for its supported modes."""
        for mode in handler.supported_modes:
            if mode in self._handlers:
                raise ValueError(f"Handler for mode {mode} already registered")
            self._handlers[mode] = handler
    
    def get_handler(self, mode: Mode) -> BaseResponseHandler:
        """Get handler for a specific mode."""
        if mode not in self._handlers:
            raise ValueError(f"No handler registered for mode {mode}")
        return self._handlers[mode]
    
    def get_supported_modes(self) -> set[Mode]:
        """Get all supported modes."""
        return set(self._handlers.keys())


# Auto-register all handlers
def create_default_registry() -> HandlerRegistry:
    """Create and populate the default handler registry."""
    registry = HandlerRegistry()
    
    # OpenAI handlers
    from .openai.tools import OpenAIToolsHandler, OpenAIToolsStrictHandler
    from .openai.json import OpenAIJsonHandler, OpenAIJsonO1Handler
    
    registry.register(OpenAIToolsHandler())
    registry.register(OpenAIToolsStrictHandler())
    registry.register(OpenAIJsonHandler())
    registry.register(OpenAIJsonO1Handler())
    
    # Anthropic handlers
    from .anthropic.tools import AnthropicToolsHandler
    from .anthropic.json import AnthropicJsonHandler
    
    registry.register(AnthropicToolsHandler())
    registry.register(AnthropicJsonHandler())
    
    # ... register all other handlers
    
    return registry

# Global registry instance
_default_registry = create_default_registry()
```

### 5. Simplify Main Handler Function

```python
# instructor/process_response.py (simplified)
from .handlers.registry import _default_registry

def handle_response_model(
    response_model: type[T] | None, 
    mode: Mode = Mode.TOOLS, 
    **kwargs: Any
) -> tuple[type[T] | VertexAIParallelBase | None, dict[str, Any]]:
    """
    Simplified handler function using the registry pattern.
    """
    new_kwargs = kwargs.copy()
    autodetect_images = new_kwargs.pop("autodetect_images", False)
    
    if response_model is None:
        return _handle_no_response_model(mode, new_kwargs)
    
    # Handle parallel modes separately (they have different return types)
    if mode in {Mode.PARALLEL_TOOLS, Mode.VERTEXAI_PARALLEL_TOOLS}:
        return _handle_parallel_modes(response_model, mode, new_kwargs)
    
    response_model = prepare_response_model(response_model)
    
    # Use the registry to get the appropriate handler
    handler = _default_registry.get_handler(mode)
    response_model, new_kwargs = handler.handle(response_model, new_kwargs)
    
    # Post-processing (common to all handlers)
    if "messages" in new_kwargs:
        new_kwargs["messages"] = convert_messages(
            new_kwargs["messages"], mode, autodetect_images=autodetect_images
        )
    
    return response_model, new_kwargs
```

### 6. Add Handler Composition and Middleware

```python
# instructor/handlers/middleware.py
class HandlerMiddleware(ABC):
    """Base class for handler middleware."""
    
    @abstractmethod
    def process_request(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        """Process before main handler."""
        pass
    
    @abstractmethod
    def process_response(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        """Process after main handler."""
        pass


class MessageConversionMiddleware(HandlerMiddleware):
    """Middleware for message format conversion."""
    
    def process_response(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        if "messages" in kwargs:
            kwargs["messages"] = convert_messages(kwargs["messages"], self.mode)
        return response_model, kwargs


class CompositeHandler(BaseResponseHandler):
    """Handler that composes multiple handlers and middleware."""
    
    def __init__(self, handler: BaseResponseHandler, middleware: List[HandlerMiddleware] = None):
        self.handler = handler
        self.middleware = middleware or []
    
    def handle(self, response_model: type[T], kwargs: Dict[str, Any]) -> Tuple[type[T], Dict[str, Any]]:
        # Apply pre-processing middleware
        for middleware in self.middleware:
            response_model, kwargs = middleware.process_request(response_model, kwargs)
        
        # Run main handler
        response_model, kwargs = self.handler.handle(response_model, kwargs)
        
        # Apply post-processing middleware
        for middleware in reversed(self.middleware):
            response_model, kwargs = middleware.process_response(response_model, kwargs)
        
        return response_model, kwargs
```

## Benefits of This Refactor

### 1. **Improved Maintainability**
- Each handler is focused on a single responsibility
- Provider-specific logic is isolated
- Easy to locate and modify specific behavior

### 2. **Better Testability**
- Each handler can be unit tested in isolation
- Mock dependencies easily
- Test coverage becomes more granular

### 3. **Enhanced Extensibility**
- Adding new providers requires no changes to core logic
- New handlers implement well-defined interfaces
- Plugin-like architecture for easy extension

### 4. **Reduced Code Duplication**
- Common patterns extracted to base classes
- Shared utilities in dedicated modules
- Template method pattern for similar workflows

### 5. **Better Organization**
- Logical grouping by provider
- Clear separation of concerns
- Easier onboarding for new contributors

### 6. **Type Safety**
- Better type hints and validation
- Compile-time checks for handler compatibility
- Clearer interfaces and contracts

## Migration Strategy

### Phase 1: Extract Base Classes
1. Create `instructor/handlers/base.py` with abstract base classes
2. Create registry system
3. Maintain backward compatibility

### Phase 2: Migrate Handlers Gradually
1. Start with OpenAI handlers (most commonly used)
2. Move one provider at a time
3. Keep old functions as wrappers initially

### Phase 3: Add Advanced Features
1. Implement middleware system
2. Add handler composition
3. Add validation and error handling improvements

### Phase 4: Cleanup
1. Remove old handler functions
2. Update documentation
3. Add comprehensive tests

## Example Usage After Refactor

```python
# Custom handler for a new provider
class CustomProviderHandler(ToolBasedHandler):
    @property
    def supported_modes(self) -> set[Mode]:
        return {Mode.CUSTOM_TOOLS}
    
    def _prepare_tools(self, response_model, kwargs):
        # Custom tool preparation logic
        return kwargs
    
    def _set_tool_choice(self, response_model, kwargs):
        # Custom tool choice logic
        return kwargs

# Register the handler
from instructor.handlers.registry import _default_registry
_default_registry.register(CustomProviderHandler())

# Usage remains the same
client = instructor.from_openai(openai_client)
result = client.chat.completions.create(
    model="custom-model",
    response_model=MyModel,
    mode=Mode.CUSTOM_TOOLS,
    messages=[...]
)
```

This refactor provides a clean, extensible, and maintainable architecture while preserving the existing API surface.