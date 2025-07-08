"""OpenAI response processing utilities."""

from typing import Any, TypeVar
from pydantic import BaseModel
from openai import pydantic_function_tool

from ...mode import Mode
from ...dsl.parallel import ParallelModel, handle_parallel_model

T = TypeVar("T")
T_Model = TypeVar("T_Model", bound=BaseModel)

def handle_parallel_tools(
    response_model: type[T],
    new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    """Handle parallel tools mode.
    
    Args:
        response_model: Expected response model
        new_kwargs: Request kwargs
        
    Returns:
        Tuple of (response model, updated kwargs)
        
    Raises:
        ConfigurationError: If streaming is enabled
    """
    if new_kwargs.get("stream", False):
        from instructor.exceptions import ConfigurationError
        raise ConfigurationError(
            "stream=True is not supported when using PARALLEL_TOOLS mode"
        )
        
    new_kwargs["tools"] = handle_parallel_model(response_model)
    new_kwargs["tool_choice"] = "auto"
    return ParallelModel(typehint=response_model), new_kwargs

def handle_functions(
    response_model: type[T],
    new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    """Handle functions mode.
    
    Args:
        response_model: Expected response model
        new_kwargs: Request kwargs
        
    Returns:
        Tuple of (response model, updated kwargs)
    """
    Mode.warn_mode_functions_deprecation()
    new_kwargs["functions"] = [response_model.openai_schema]
    new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}
    return response_model, new_kwargs

def handle_tools_strict(
    response_model: type[T],
    new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    """Handle strict tools mode.
    
    Args:
        response_model: Expected response model
        new_kwargs: Request kwargs
        
    Returns:
        Tuple of (response model, updated kwargs)
    """
    response_model_schema = pydantic_function_tool(response_model)
    response_model_schema["function"]["strict"] = True
    new_kwargs["tools"] = [response_model_schema]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model_schema["function"]["name"]},
    }
    return response_model, new_kwargs

def handle_tools(
    response_model: type[T],
    new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    """Handle tools mode.
    
    Args:
        response_model: Expected response model
        new_kwargs: Request kwargs
        
    Returns:
        Tuple of (response model, updated kwargs)
    """
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model.openai_schema["name"]},
    }
    return response_model, new_kwargs

def handle_json_modes(
    response_model: type[T],
    new_kwargs: dict[str, Any],
    mode: Mode
) -> tuple[type[T], dict[str, Any]]:
    """Handle JSON modes.
    
    Args:
        response_model: Expected response model
        new_kwargs: Request kwargs
        mode: Request mode
        
    Returns:
        Tuple of (response model, updated kwargs)
    """
    if mode == Mode.JSON:
        new_kwargs["response_format"] = {"type": "json_object"}
    elif mode == Mode.JSON_SCHEMA:
        new_kwargs["response_format"] = {
            "type": "json_object",
            "schema": response_model.model_json_schema()
        }
    return response_model, new_kwargs 