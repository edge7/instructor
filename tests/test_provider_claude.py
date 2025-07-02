import types
import sys
import pytest
import anyio
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Stub out `claude_code_sdk` so that the provider can be imported without the
# real package present. This needs to be done *before* importing the provider.
# ---------------------------------------------------------------------------

class _DummyMessage:
    """Simple stand-in for the SDK's streamed Message objects."""

    def __init__(self, kind: str, text: str):
        self.kind = kind
        self.text = text


async def _mock_query(*args, **kwargs):  # noqa: D401 – simple mock
    """Yield a single dummy assistant chunk containing *kwargs['mock_text']*."""
    # The test sets `mock_text` as kwarg to control output
    text = kwargs.pop("mock_text", "")
    yield _DummyMessage(kind="assistant", text=text)


# Build the fake module with the attributes expected by provider_claude.py
mock_sdk = types.ModuleType("claude_code_sdk")
mock_sdk.query = _mock_query  # type: ignore[attr-defined]

# Minimal placeholder for options – the provider never introspects attributes
class _DummyOptions:  # noqa: D401 – simple dummy
    def __init__(self, **_):
        pass

mock_sdk.ClaudeCodeOptions = _DummyOptions  # type: ignore[attr-defined]
mock_sdk.Message = _DummyMessage  # type: ignore[attr-defined]

sys.modules["claude_code_sdk"] = mock_sdk

# ---------------------------------------------------------------------------
# Now we can safely import the provider under test
# ---------------------------------------------------------------------------
from provider_claude import ClaudeCodeProvider  # noqa: E402  – after stubbing


class _Haiku(BaseModel):
    title: str
    poem: str
    author: str


@pytest.mark.anyio
async def test_acompletion_raw_text():
    provider = ClaudeCodeProvider()
    result = await provider.acompletion(
        messages=[{"role": "user", "content": "Hello"}],
        options=_DummyOptions(),
        mock_text="Hello world!",
    )
    assert result == "Hello world!"


@pytest.mark.anyio
async def test_acompletion_json_parsing():
    provider = ClaudeCodeProvider()
    # Build a JSON string that conforms to _Haiku
    json_payload = '```json\n{"title": "foo", "poem": "bar", "author": "baz"}\n```'
    res = await provider.acompletion(
        messages=[{"role": "user", "content": "Write"}],
        response_model=_Haiku,
        options=_DummyOptions(),
        mock_text=json_payload,
    )
    assert isinstance(res, _Haiku)
    assert res.title == "foo"
    assert res.poem == "bar"
    assert res.author == "baz"