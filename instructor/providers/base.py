from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Union, TYPE_CHECKING
from collections.abc import Awaitable

if TYPE_CHECKING:
    import instructor
    from instructor.hooks import Hooks
    from instructor.utils import Provider
    ChatCompletionMessageParam = Any

T = TypeVar("T")


class BaseProvider(ABC):
    """
    Abstract base class for all provider implementations.
    
    This class defines the interface that all providers must implement
    to integrate with the instructor library.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name identifier for this provider."""
        pass
    
    @abstractmethod
    def validate_client(self, client: Any) -> bool:
        """
        Validate that the client is compatible with this provider.
        
        Args:
            client: The client object to validate
            
        Returns:
            bool: True if the client is valid for this provider
        """
        pass
    
    @abstractmethod
    def get_create_function(
        self, 
        client: Any, 
        mode: Any  # instructor.Mode
    ) -> Callable[..., Any]:
        """
        Get the create function for this provider.
        
        Args:
            client: The client object
            mode: The instructor mode to use
            
        Returns:
            The create function to use for making API calls
        """
        pass
    
    @abstractmethod
    def get_instructor_class(self, client: Any) -> type[Any]:  # type[instructor.Instructor] | type[instructor.AsyncInstructor]
        """
        Get the appropriate Instructor class for this provider.
        
        Args:
            client: The client object
            
        Returns:
            The Instructor or AsyncInstructor class to use
        """
        pass
    
    @abstractmethod
    def get_provider_enum(self) -> Any:  # Provider
        """
        Get the Provider enum value for this provider.
        
        Returns:
            The Provider enum value
        """
        pass
    
    @abstractmethod
    def validate_mode(self, mode: Any) -> bool:  # mode: instructor.Mode
        """
        Validate that the mode is supported by this provider.
        
        Args:
            mode: The instructor mode to validate
            
        Returns:
            bool: True if the mode is supported
        """
        pass
    
    def create_instructor(
        self,
        client: Any,
        mode: Any = None,  # instructor.Mode = instructor.Mode.TOOLS,
        hooks: Any | None = None,  # Hooks | None = None,
        **kwargs: Any,
    ) -> Any:  # instructor.Instructor | instructor.AsyncInstructor
        """
        Create an Instructor instance for this provider.
        
        Args:
            client: The client object
            mode: The instructor mode to use
            hooks: Optional hooks to attach
            **kwargs: Additional keyword arguments
            
        Returns:
            An Instructor or AsyncInstructor instance
        """
        if not self.validate_client(client):
            raise ValueError(f"Invalid client for {self.provider_name} provider")
        
        if not self.validate_mode(mode):
            raise ValueError(f"Mode {mode} is not supported by {self.provider_name} provider")
        
        instructor_class = self.get_instructor_class(client)
        create_function = self.get_create_function(client, mode)
        provider_enum = self.get_provider_enum()
        
        return instructor_class(
            client=client,
            create=create_function,
            mode=mode,
            provider=provider_enum,
            hooks=hooks,
            **kwargs,
        )


class ProviderRegistry:
    """Registry for managing provider implementations."""
    
    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
    
    def register(self, provider: BaseProvider) -> None:
        """Register a provider implementation."""
        self._providers[provider.provider_name] = provider
    
    def get_provider(self, name: str) -> BaseProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)
    
    def get_provider_for_client(self, client: Any) -> BaseProvider | None:
        """Find the appropriate provider for a given client."""
        for provider in self._providers.values():
            if provider.validate_client(client):
                return provider
        return None
    
    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())


# Global registry instance
registry = ProviderRegistry()