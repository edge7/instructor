from __future__ import annotations

"""Base abstractions for LLM providers.

This module purposely lives outside of *instructor* so that we can grow a
plug-in architecture without introducing circular imports.  Nothing in here
imports heavy optional provider libraries such as *openai* or *anthropic* –
sub-modules that implement concrete providers should keep those imports behind
`try/except ImportError` guards so that users do **not** have to install every
single provider dependency.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

__all__ = [
    "BaseProvider",
    "ProviderRegistry",
    "register_provider",
]


class ProviderRegistry:
    """A tiny registry mapping provider names to their concrete classes."""

    _providers: Dict[str, Type["BaseProvider"]] = {}

    @classmethod
    def register(cls, provider_cls: Type["BaseProvider"]) -> Type["BaseProvider"]:
        """Register *provider_cls* under its declared :pyattr:`name`.

        Implementations must declare a **unique** ``name`` class attribute.  We
        intentionally do **not** validate uniqueness – if two providers use the
        same name the latter silently wins – mirroring how most plug-in
        registries work and keeping the logic minimal.
        """
        name = getattr(provider_cls, "name", None)
        if not name:
            raise ValueError("Provider classes must define a `name` class attribute.")
        cls._providers[name] = provider_cls
        return provider_cls

    # Convenience helpers -------------------------------------------------
    @classmethod
    def get(cls, name: str) -> Type["BaseProvider"] | None:  # pragma: no cover
        """Return the concrete provider class registered under *name* (if any)."""
        return cls._providers.get(name)

    @classmethod
    def available(cls) -> Dict[str, Type["BaseProvider"]]:  # pragma: no cover
        """Return a mapping of all registered providers."""
        return dict(cls._providers)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class BaseProvider(ABC):
    """Common interface that every provider implementation must follow."""

    #: Short identifier used when registering the provider.
    name: str = ""

    # NOTE: We purposefully avoid importing *Instructor* types here to keep this
    # module dependency-free.  Sub-classes can annotate the return value more
    # concretely.
    @classmethod
    @abstractmethod
    def from_client(cls, client: Any, **kwargs: Any):  # noqa: D401,E501
        """Return an :pyclass:`instructor.client.Instructor` instance for *client*.

        The concrete return type is kept generic (``Any``) to avoid heavy
        imports.  Users are expected to know the exact type or to rely on type
        checkers that understand :pep:`604` unions.
        """


# ---------------------------------------------------------------------------
# Registration decorator – purely syntactic sugar
# ---------------------------------------------------------------------------

def register_provider(provider_cls: Type[BaseProvider]) -> Type[BaseProvider]:
    """Class decorator that registers *provider_cls* in :class:`ProviderRegistry`."""

    return ProviderRegistry.register(provider_cls)