"""
promptframe - A lightweight bridge between Pandas rows and instructor-powered, 
Jinja-templated, structured-LLM extraction.

A pandas-native, type-safe, retry-aware, vectorised path for adding LLM-derived 
features to tabular data.
"""

from .core import PromptFrame
from .types import PromptFrameError

__version__ = "0.1.0"
__all__ = ["PromptFrame", "PromptFrameError"]