from typing import Any

from ..base import BaseProvider, Mode


class ProviderRegistry:
    """Registry for managing and looking up LLM providers.

    This class provides a centralized way to register providers and look them up
    by name. It also handles validation of provider capabilities.
    """

    _providers: dict[str, type[BaseProvider]] = {}

    @classmethod
    def register(cls, provider_class: type[BaseProvider]) -> type[BaseProvider]:
        """Register a provider class.

        This can be used as a decorator:

        @ProviderRegistry.register
        class MyProvider(BaseProvider):
            ...

        Args:
            provider_class: The provider class to register

        Returns:
            The registered provider class

        Raises:
            ValueError: If a provider with the same name is already registered
        """
        if not hasattr(provider_class, "name"):
            raise ValueError(
                f"Provider class {provider_class.__name__} must define a 'name' attribute"
            )

        if not hasattr(provider_class, "supported_modes"):
            raise ValueError(
                f"Provider class {provider_class.__name__} must define 'supported_modes'"
            )

        if provider_class.name in cls._providers:
            raise ValueError(f"Provider '{provider_class.name}' is already registered")

        cls._providers[provider_class.name] = provider_class
        return provider_class

    @classmethod
    def get_provider(cls, name: str) -> type[BaseProvider]:
        """Look up a provider by name.

        Args:
            name: The name of the provider to look up

        Returns:
            The provider class

        Raises:
            KeyError: If no provider is registered with the given name
        """
        if name not in cls._providers:
            raise KeyError(f"No provider registered with name '{name}'")
        return cls._providers[name]

    @classmethod
    def validate_provider(
        cls, provider: BaseProvider, required_modes: set[Mode]
    ) -> None:
        """Validate that a provider supports the required modes.

        Args:
            provider: The provider instance to validate
            required_modes: Set of modes that the provider must support

        Raises:
            ValueError: If the provider doesn't support all required modes
        """
        unsupported_modes = required_modes - provider.supported_modes
        if unsupported_modes:
            raise ValueError(
                f"Provider '{provider.name}' does not support the following modes: "
                f"{', '.join(unsupported_modes)}"
            )
