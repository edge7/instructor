"""Type definitions and exceptions for promptframe."""

from typing import Any, Dict, List, Optional, Union, Callable, Type
from pydantic import BaseModel
import pandas as pd


class PromptFrameError(Exception):
    """Base exception for promptframe errors."""
    pass


class ValidationError(PromptFrameError):
    """Raised when Pydantic validation fails."""
    pass


class TemplateError(PromptFrameError):
    """Raised when Jinja template rendering fails."""
    pass


class LLMError(PromptFrameError):
    """Raised when LLM API calls fail."""
    pass


# Type aliases for clarity
TemplateType = Union[str, Callable[[pd.Series], str]]
OptionalTemplateType = Union[str, Callable[[pd.Series], str], None]
TemplateKwargsType = Union[Dict[str, Any], Callable[[pd.Series], Dict[str, Any]], None]
SchemaType = Type[BaseModel]