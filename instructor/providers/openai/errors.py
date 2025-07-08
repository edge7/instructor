"""OpenAI error handling module."""

from typing import Any, Optional
from openai.types.chat import ChatCompletion
from openai import (
    APIError,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from ...exceptions import (
    InstructorError,
    ValidationError,
    RetryError,
    RateLimitError as InstructorRateLimitError,
    TimeoutError as InstructorTimeoutError,
    APIError as InstructorAPIError,
)


def handle_openai_error(
    error: Exception, response: Optional[ChatCompletion] = None, **kwargs: Any
) -> None:
    """Handle OpenAI API errors.

    Args:
        error: The error that occurred
        response: Raw API response
        **kwargs: Additional error handling options

    Raises:
        InstructorError: Re-raised with appropriate context
    """
    # Handle validation errors
    if isinstance(error, ValidationError):
        raise InstructorError(
            message="Response validation failed",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle rate limit errors
    if isinstance(error, RateLimitError):
        raise InstructorRateLimitError(
            message="OpenAI API rate limit exceeded",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle timeout errors
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        raise InstructorTimeoutError(
            message="OpenAI API request timed out",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle authentication errors
    if isinstance(error, AuthenticationError):
        raise InstructorAPIError(
            message="OpenAI API authentication failed",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle permission errors
    if isinstance(error, PermissionDeniedError):
        raise InstructorAPIError(
            message="OpenAI API permission denied",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle bad request errors
    if isinstance(error, BadRequestError):
        raise InstructorAPIError(
            message="OpenAI API bad request",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle not found errors
    if isinstance(error, NotFoundError):
        raise InstructorAPIError(
            message="OpenAI API resource not found",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle conflict errors
    if isinstance(error, ConflictError):
        raise InstructorAPIError(
            message="OpenAI API conflict",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle unprocessable entity errors
    if isinstance(error, UnprocessableEntityError):
        raise InstructorAPIError(
            message="OpenAI API unprocessable entity",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle internal server errors
    if isinstance(error, InternalServerError):
        raise RetryError(
            message="OpenAI API internal server error",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle generic API errors
    if isinstance(error, APIError):
        raise InstructorAPIError(
            message="OpenAI API error",
            response=response,
            original_error=error,
            **kwargs,
        ) from error

    # Handle unknown errors
    raise InstructorError(
        message="Unknown error occurred",
        response=response,
        original_error=error,
        **kwargs,
    ) from error
