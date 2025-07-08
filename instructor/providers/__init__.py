"""Provider implementations."""

from typing import Dict, Type

from .base import BaseProvider

# Registry of available providers
PROVIDERS: dict[str, type[BaseProvider]] = {}
