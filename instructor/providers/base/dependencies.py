"""Provider dependency management utilities."""

import functools
import importlib.util
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

def requires_package(package_name: str, min_version: Optional[str] = None) -> Callable[[T], T]:
    """Decorator to ensure required packages are installed.
    
    Args:
        package_name: Name of the required package
        min_version: Minimum version required (optional)
        
    Returns:
        Decorator function that checks package requirements
        
    Example:
        @requires_package("openai", min_version="1.0.0")
        class OpenAIProvider:
            ...
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if package is installed
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                raise ImportError(
                    f"Package '{package_name}' is required but not installed. "
                    f"Please install it with: pip install {package_name}"
                    + (f">={min_version}" if min_version else "")
                )
                
            # Check version if specified
            if min_version:
                import pkg_resources
                installed_version = pkg_resources.get_distribution(package_name).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                    raise ImportError(
                        f"Package '{package_name}' version {min_version} or higher is required, "
                        f"but version {installed_version} is installed. "
                        f"Please upgrade with: pip install {package_name}>={min_version}"
                    )
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator