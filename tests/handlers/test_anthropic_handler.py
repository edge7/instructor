import pytest
from typing import Any

from pydantic import BaseModel

import instructor
from instructor.mode import Mode
from instructor.handlers.registry import get as get_handler


class DummyModel(BaseModel):
    foo: str

    # Minimal schema attributes required by the Anthropic handler
    anthropic_schema: dict[str, Any] = {
        "name": "DummyModel",
        "parameters": {
            "type": "object",
            "properties": {"foo": {"type": "string"}},
        },
    }


@pytest.mark.parametrize("mode", [
    Mode.ANTHROPIC_JSON,
    Mode.ANTHROPIC_TOOLS,
    Mode.ANTHROPIC_REASONING_TOOLS,
])
def test_prepare_request(mode: Mode):
    handler = get_handler(mode)
    assert handler is not None, "Handler should be registered"

    call_kwargs = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
    }
    rm, new_kwargs = handler.prepare_request(mode, DummyModel, call_kwargs)

    # Ensure original kwargs are not mutated
    assert call_kwargs is not new_kwargs
    assert rm is DummyModel

    # Basic sanity checks depending on mode
    if mode is Mode.ANTHROPIC_JSON:
        assert "system" in new_kwargs, "Json mode should inject system schema"
    else:
        assert "tools" in new_kwargs, "Tools modes should include tool definitions"


def test_parse_response_passthrough():
    """parse_response should return None when no model is supplied, delegating to legacy parser."""
    mode = Mode.ANTHROPIC_JSON
    handler = get_handler(mode)
    result = handler.parse_response(  # type: ignore[attr-defined]
        mode,
        raw_completion={"dummy": 1},
        response_model=None,
        stream=False,
        validation_context=None,
        strict=None,
    )
    assert result is None