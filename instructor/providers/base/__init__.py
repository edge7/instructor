"""Base provider implementation."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Set
from ...mode import Mode


class BaseProvider(ABC):
    """Base class for all providers.
    
    This class defines the interface that all providers must implement.
    It includes methods for request preparation, response processing,
    error handling, and retry configuration.
    """
    
    name: str
    supported_modes: ClassVar[Set[Mode]]
    
    def __init__(self) -> None:
        """Initialize base provider."""
        self._retry_config: Dict[str, Any] = {
            "max_retries": 3,
            "timeout": 60,
            "conditions": set()
        }
    
    @abstractmethod
    def prepare_request(self, response_model: Any, mode: Mode, **kwargs: Any) -> dict[str, Any]:
        """Prepare request for provider API.
        
        Args:
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional request options
            
        Returns:
            Prepared request parameters
        """
        pass
    
    @abstractmethod
    def process_response(self, response: Any, response_model: Any, mode: Mode, **kwargs: Any) -> Any:
        """Process response from provider API.
        
        Args:
            response: Raw API response
            response_model: Expected response model
            mode: Request mode
            **kwargs: Additional processing options
            
        Returns:
            Processed response
        """
        pass
    
    @abstractmethod
    def handle_error(self, error: Exception, response: Any, **kwargs: Any) -> None:
        """Handle provider API errors.
        
        Args:
            error: The error that occurred
            response: Raw API response
            **kwargs: Additional error handling options
        """
        pass
    
    def configure_retry(self, max_retries: Optional[int] = None, timeout: Optional[int] = None) -> None:
        """Configure retry settings.
        
        Args:
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds between retries
        """
        if max_retries is not None:
            self._retry_config["max_retries"] = max_retries
        if timeout is not None:
            self._retry_config["timeout"] = timeout
    
    def get_retry_conditions(self) -> Set[Any]:
        """Get provider-specific retry conditions.
        
        Returns:
            Set of conditions that should trigger a retry
        """
        return self._retry_config["conditions"]
    
    def add_retry_condition(self, condition: Any) -> None:
        """Add a retry condition.
        
        Args:
            condition: Condition that should trigger a retry
        """
        self._retry_config["conditions"].add(condition)
    
    def remove_retry_condition(self, condition: Any) -> None:
        """Remove a retry condition.
        
        Args:
            condition: Condition to remove from retry triggers
        """
        self._retry_config["conditions"].discard(condition)
    
    @property
    def retry_config(self) -> Dict[str, Any]:
        """Get current retry configuration.
        
        Returns:
            Dictionary containing retry settings
        """
        return self._retry_config.copy() 