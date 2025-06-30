"""Basic tests covering the provider abstraction migration.

These tests rely on a *stub* implementation of the ``openai`` package so that
we do not need the real dependency.  They exercise both the *new* provider
path (default) and the *legacy* fallback path.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from types import ModuleType

import pytest  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_openai_stub() -> ModuleType:  # noqa: D401
    """Install a minimal stub of the *openai* module into *sys.modules*."""

    if "openai" in sys.modules:  # pragma: no cover – keep existing stub
        return sys.modules["openai"]

    openai_stub = types.ModuleType("openai")

    # ----------------------------------------------------
    # Stub *types.chat.ChatCompletionMessageParam*
    # ----------------------------------------------------
    types_mod = types.ModuleType("openai.types")
    chat_mod = types.ModuleType("openai.types.chat")
    setattr(chat_mod, "ChatCompletionMessageParam", dict)  # simplistic alias
    types_mod.chat = chat_mod  # type: ignore[attr-defined]
    sys.modules["openai.types"] = types_mod
    sys.modules["openai.types.chat"] = chat_mod

    # ----------------------------------------------------
    # Stub client classes utilised by *instructor*
    # ----------------------------------------------------
    class _StubChatCompletions:  # noqa: D401
        @staticmethod
        def create(*_args, **_kwargs):  # noqa: D401
            return {"status": "ok"}

    class _StubChat:  # noqa: D401
        completions = _StubChatCompletions()

    class _BaseClient:  # noqa: D401
        def __init__(self):
            self.chat = _StubChat()

    openai_stub.OpenAI = _BaseClient  # type: ignore[attr-defined]
    openai_stub.AsyncOpenAI = _BaseClient  # type: ignore[attr-defined]

    # provide *types* submodule container
    openai_stub.types = types_mod  # type: ignore[attr-defined]

    sys.modules["openai"] = openai_stub
    return openai_stub


@contextmanager
def _force_legacy_path():  # noqa: D401
    """Temporarily force *instructor.client.from_openai* to take the legacy path."""

    import builtins  # late import to avoid early capture

    orig_import = builtins.__import__

    def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: D401
        if name == "providers.client_openai":
            raise ModuleNotFoundError
        return orig_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _mock_import  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.__import__ = orig_import  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_provider_path(monkeypatch):
    """The happy path uses the new provider abstraction."""

    # Ensure stub is installed **before** importing *instructor.client*
    openai_stub = _install_openai_stub()
    monkeypatch.setitem(sys.modules, "openai", openai_stub)

    import instructor.client as ic  # pylint: disable=import-error

    # Reload to make sure the stub is picked up in case of import order issues
    ic = importlib.reload(ic)

    client = openai_stub.OpenAI()  # type: ignore[arg-type]
    inst = ic.from_openai(client)

    # Import the provider directly to cross-check the factory
    from providers.client_openai import OpenAIProvider  # type: ignore  # pylint: disable=import-error

    expected = OpenAIProvider.from_client(client)

    assert isinstance(inst, type(expected))


def test_legacy_fallback(monkeypatch):
    """Import failure of provider triggers legacy fallback code path."""

    openai_stub = _install_openai_stub()
    monkeypatch.setitem(sys.modules, "openai", openai_stub)

    with _force_legacy_path():
        import instructor.client as ic  # pylint: disable=import-error

        ic = importlib.reload(ic)
        client = openai_stub.OpenAI()  # type: ignore[arg-type]

        inst = ic.from_openai(client)

        # Sanity – we got an Instructor/AsyncInstructor object back
        from instructor.client import Instructor, AsyncInstructor  # pylint: disable=import-error

        assert isinstance(inst, (Instructor, AsyncInstructor))