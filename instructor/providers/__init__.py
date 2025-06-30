"""Provider abstraction for instructor library."""

from .base import BaseProvider, ProviderRegistry, registry

__all__ = [
    "BaseProvider",
    "ProviderRegistry", 
    "registry",
]

# Conditionally import providers based on available dependencies
# Note: We don't auto-register providers in __init__ to avoid import issues
# Providers should be imported explicitly when needed