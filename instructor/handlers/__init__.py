from __future__ import annotations

from .registry import get, register  # noqa: F401

# Ensure built-in handlers are registered on import
from . import anthropic  # noqa: F401