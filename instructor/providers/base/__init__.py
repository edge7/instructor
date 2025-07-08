from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

class Mode(str):
    """Represents different modes of operation for providers."""
    FUNCTIONS = "functions"
    TOOLS = "tools"
    TOOLS_STRICT = "tools-strict"
    JSON = "json"
    JSON_STRICT = "json-strict"
    PARALLEL_TOOLS = "parallel-tools"

T = TypeVar("T", bound=BaseModel)

class BaseProvider(ABC):
    """Base class for all LLM providers.
    
    This class defines the interface that all providers must implement to be compatible
    with the instructor system. Each provider handles its own request preparation,
    response processing, and error handling.
    """
    
    name: str
    """The name of the provider."""
    
    supported_modes: ClassVar[set[Mode]]
    """The set of modes supported by this provider."""
    
    @abstractmethod
    def prepare_request(
        self,
        response_model: type[T],
        mode: Mode,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare the request payload for the provider.
        
        Args:
            response_model: The Pydantic model class that defines the expected response structure
            mode: The mode of operation (e.g., functions, tools, json)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            A dictionary containing the prepared request payload
        """
        pass
    
    @abstractmethod
    def process_response(
        self,
        response: Any,
        response_model: type[T],
        mode: Mode,
        **kwargs: Any
    ) -> T:
        """Process the provider's response into the expected model.
        
        Args:
            response: The raw response from the provider
            response_model: The Pydantic model class to parse the response into
            mode: The mode of operation used for the request
            **kwargs: Additional provider-specific arguments
            
        Returns:
            An instance of the response model populated with the processed data
        """
        pass
    
    @abstractmethod
    def handle_error(
        self,
        error: Exception,
        response: Any,
        **kwargs: Any
    ) -> None:
        """Handle provider-specific errors.
        
        Args:
            error: The exception that was raised
            response: The raw response that caused the error (if any)
            **kwargs: Additional provider-specific arguments
            
        Raises:
            Exception: Re-raises the error after any provider-specific handling
        """
        pass 