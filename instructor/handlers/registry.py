from __future__ import annotations

from typing import Dict

from instructor.mode import Mode

from .base import ResponseHandler

# Internal registry mapping Mode -> handler instance
_registry: Dict[Mode, ResponseHandler] = {}


def register(handler: ResponseHandler) -> ResponseHandler:
    """Register a handler instance for all its supported modes."""
    for mode in handler.supported_modes:
        if mode in _registry:
            # Later registrations override previous ones to make overriding easy
            # but emit a debug log so users are aware
            try:
                import logging

                logging.getLogger("instructor.handlers").debug(
                    "Overriding existing handler for mode %s with %s",
                    mode,
                    handler.__class__.__name__,
                )
            except Exception:
                pass
        _registry[mode] = handler
    return handler


def get(mode: Mode):
    """Return a handler instance for the given mode or None."""
    return _registry.get(mode)