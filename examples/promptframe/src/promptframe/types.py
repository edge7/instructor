"""Type definitions and exceptions for promptframe."""

from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel


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
TemplateKwargsType = Union[
    Dict[str, Any], Callable[[pd.Series], Dict[str, Any]], None
]
SchemaType = Type[BaseModel]