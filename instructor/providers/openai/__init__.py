"""OpenAI provider implementation."""

from typing import Any, ClassVar, TypeVar
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from ...mode import Mode
from ..base import BaseProvider
from ..base.dependencies import requires_package
from ...function_calls import OpenAISchema
from ...dsl.iterable import IterableBase
from ...dsl.partial import PartialBase
from .response import (
    handle_functions,
    handle_tools,
    handle_tools_strict,
    handle_parallel_tools,
    handle_json_modes,
)
from .errors import reask_tools

T = TypeVar("T")
T_Model = TypeVar("T_Model", bound=BaseModel)

@requires_package("openai", min_version="1.0.0")
class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation.
    
    This provider handles all OpenAI-specific functionality including:
    - Request preparation for different modes (functions, tools, json)
    - Response processing and validation
    - Error handling and retry logic
    """
    
    name: str = "openai"
    supported_modes: ClassVar[set[Mode]] = {
        Mode.FUNCTIONS,
        Mode.TOOLS,
        Mode.TOOLS_STRICT,
        Mode.JSON,
        Mode.JSON_SCHEMA,
        Mode.PARALLEL_TOOLS,
    }
    
    def prepare_request(
        self,
        response_model: type[T_Model | OpenAISchema | BaseModel] | None,
        mode: Mode,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare request for OpenAI API.
        
        Args:
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional request options
            
        Returns:
            Prepared request parameters
        """
        if response_model is None:
            return kwargs
            
        new_kwargs = kwargs.copy()
        
        # Handle different modes
        if mode == Mode.FUNCTIONS:
            response_model, new_kwargs = handle_functions(response_model, new_kwargs)
        elif mode == Mode.TOOLS:
            response_model, new_kwargs = handle_tools(response_model, new_kwargs)
        elif mode == Mode.TOOLS_STRICT:
            response_model, new_kwargs = handle_tools_strict(response_model, new_kwargs)
        elif mode == Mode.PARALLEL_TOOLS:
            response_model, new_kwargs = handle_parallel_tools(response_model, new_kwargs)
        elif mode in {Mode.JSON, Mode.JSON_SCHEMA}:
            response_model, new_kwargs = handle_json_modes(response_model, new_kwargs, mode)
            
        return new_kwargs
        
    def process_response(
        self,
        response: ChatCompletion,
        response_model: type[T_Model | OpenAISchema | BaseModel] | None,
        mode: Mode,
        **kwargs: Any
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
        if response_model is None:
            return response
            
        validation_context = kwargs.get("validation_context")
        strict = kwargs.get("strict")
        stream = kwargs.get("stream", False)
        
        if (
            issubclass(response_model, (IterableBase, PartialBase))
            and stream
        ):
            model = response_model.from_streaming_response(
                response,
                mode=mode,
            )
            return model
            
        model = response_model.from_response(
            response,
            validation_context=validation_context,
            strict=strict,
            mode=mode,
        )
        
        model._raw_response = response
        return model
        
    def handle_error(
        self,
        error: Exception,
        response: ChatCompletion,
        **kwargs: Any
    ) -> None:
        """Handle OpenAI API errors.
        
        Args:
            error: The error that occurred
            response: Raw API response
            **kwargs: Additional error handling options
        """
        if isinstance(error, ValidationError):
            # Add validation error retry condition
            self.add_retry_condition(ValidationError)
            
            # Update kwargs for retry
            kwargs = reask_tools(kwargs, response, error)
            
            # Re-raise error with updated kwargs
            error.kwargs = kwargs
            raise error