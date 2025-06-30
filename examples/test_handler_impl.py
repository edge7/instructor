"""Minimal playground to showcase the new handler architecture.

Run:

```bash
python examples/test_handler_impl.py
```

No network calls are made – we only exercise the *prepare_request* step of the
`AnthropicHandler` so you can see how it mutates the kwargs per‐mode.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from pydantic import BaseModel  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("pydantic must be installed to run this example")

from instructor.mode import Mode
from instructor.handlers.registry import get as get_handler


class DummyModel(BaseModel):
    """Simplest possible response model for demonstration purposes."""

    foo: str

    # Minimal fields consumed by the Anthropic handler – in production these
    # are generated automatically by `openai_schema()`.
    anthropic_schema: dict[str, Any] = {
        "name": "DummyModel",
        "parameters": {
            "type": "object",
            "properties": {"foo": {"type": "string"}},
        },
    }


# Fake chat messages from the caller
INITIAL_KWARGS: dict[str, Any] = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "return JSON with foo='bar'"},
    ]
}


def main() -> None:
    for mode in (
        Mode.ANTHROPIC_JSON,
        Mode.ANTHROPIC_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
    ):
        handler = get_handler(mode)
        if handler is None:
            raise RuntimeError(f"No handler registered for {mode}")

        # Always work on a copy so we can re-use INITIAL_KWARGS for each mode
        response_model, patched_kwargs = handler.prepare_request(  # type: ignore[arg-type]
            mode, DummyModel, INITIAL_KWARGS.copy()
        )

        print("\n" + "=" * 60)
        print(f"Mode: {mode.value}")
        print("Prepared kwargs →")
        print(json.dumps(patched_kwargs, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()