"""OpenAI response processing module."""

from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
import json

from ...mode import Mode


def process_openai_response(
    response: ChatCompletion,
    response_model: Optional[type[BaseModel]],
    mode: Mode,
    validation_context: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
    strict: bool = True,
    partial: bool = False,
    iterable: bool = False,
    **kwargs: Any,
) -> Any:
    """Process response from OpenAI API.

    Args:
        response: Raw API response
        response_model: Expected response model
        mode: Request mode
        validation_context: Context for model validation
        context: Additional context for processing
        strict: Whether to use strict validation
        partial: Whether this is a partial response
        iterable: Whether this is an iterable response
        **kwargs: Additional processing options

    Returns:
        Processed response
    """
    if response_model is None:
        return response

    # Get message content
    message = response.choices[0].message

    # Handle different modes
    if mode == Mode.FUNCTIONS:
        return _handle_functions_mode(
            message, response_model, validation_context, strict
        )
    elif mode == Mode.TOOLS:
        return _handle_tools_mode(message, response_model, validation_context, strict)
    elif mode == Mode.TOOLS_STRICT:
        return _handle_tools_strict_mode(
            message, response_model, validation_context, strict
        )
    elif mode == Mode.JSON:
        return _handle_json_mode(message, response_model, validation_context, strict)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _handle_functions_mode(
    message: ChatCompletionMessage,
    response_model: type[BaseModel],
    validation_context: Optional[dict[str, Any]] = None,
    strict: bool = True,
) -> BaseModel:
    """Handle function call mode response.

    Args:
        message: Message from API response
        response_model: Expected response model
        validation_context: Context for model validation
        strict: Whether to use strict validation

    Returns:
        Validated model instance
    """
    if not message.function_call:
        raise ValueError("No function call in response")

    try:
        # Parse function arguments
        args = json.loads(message.function_call.arguments)

        # Create and validate model
        if validation_context:
            return response_model(**args, **validation_context)
        return response_model(**args)
    except (json.JSONDecodeError, ValidationError) as e:
        if strict:
            raise
        return None


def _handle_tools_mode(
    message: ChatCompletionMessage,
    response_model: type[BaseModel],
    validation_context: Optional[dict[str, Any]] = None,
    strict: bool = True,
) -> BaseModel:
    """Handle tools mode response.

    Args:
        message: Message from API response
        response_model: Expected response model
        validation_context: Context for model validation
        strict: Whether to use strict validation

    Returns:
        Validated model instance
    """
    if not message.tool_calls:
        raise ValueError("No tool calls in response")

    try:
        # Parse tool arguments
        args = json.loads(message.tool_calls[0].function.arguments)

        # Create and validate model
        if validation_context:
            return response_model(**args, **validation_context)
        return response_model(**args)
    except (json.JSONDecodeError, ValidationError) as e:
        if strict:
            raise
        return None


def _handle_tools_strict_mode(
    message: ChatCompletionMessage,
    response_model: type[BaseModel],
    validation_context: Optional[dict[str, Any]] = None,
    strict: bool = True,
) -> BaseModel:
    """Handle strict tools mode response.

    Args:
        message: Message from API response
        response_model: Expected response model
        validation_context: Context for model validation
        strict: Whether to use strict validation

    Returns:
        Validated model instance
    """
    if not message.tool_calls:
        raise ValueError("No tool calls in response")

    try:
        # Parse tool arguments
        args = json.loads(message.tool_calls[0].function.arguments)

        # Create and validate model
        if validation_context:
            return response_model(**args, **validation_context)
        return response_model(**args)
    except (json.JSONDecodeError, ValidationError) as e:
        if strict:
            raise
        return None


def _handle_json_mode(
    message: ChatCompletionMessage,
    response_model: type[BaseModel],
    validation_context: Optional[dict[str, Any]] = None,
    strict: bool = True,
) -> BaseModel:
    """Handle JSON mode response.

    Args:
        message: Message from API response
        response_model: Expected response model
        validation_context: Context for model validation
        strict: Whether to use strict validation

    Returns:
        Validated model instance
    """
    if not message.content:
        raise ValueError("No content in response")

    try:
        # Parse JSON content
        data = json.loads(message.content)

        # Create and validate model
        if validation_context:
            return response_model(**data, **validation_context)
        return response_model(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        if strict:
            raise
        return None
