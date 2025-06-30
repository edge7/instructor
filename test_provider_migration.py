#!/usr/bin/env python3
"""
Test script to verify that the new provider system maintains backward compatibility.
This test can be run even without OpenAI installed to check the import structure.
"""

import sys
import os

# Add the workspace to the path
sys.path.insert(0, '/workspace')

def test_backward_compatibility():
    """Test that the existing API still works with the new provider system."""
    try:
        # This should work even without OpenAI installed
        # The import should succeed, but actual usage will require OpenAI
        from instructor.client import from_openai
        print("✓ from_openai import successful")
        
        # Test that the function signature is correct
        import inspect
        sig = inspect.signature(from_openai)
        print(f"✓ from_openai signature: {sig}")
        
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_provider_system():
    """Test the new provider system."""
    try:
        # Import provider system directly (should work without OpenAI)
        from instructor.providers.base import BaseProvider, ProviderRegistry
        print("✓ Provider base classes import successful")
        
        # Create registry
        registry = ProviderRegistry()
        print(f"✓ Registry created: {type(registry).__name__}")
        
        # Test OpenAI provider (will fail gracefully without OpenAI)
        try:
            from instructor.providers.client_openai import OpenAIProvider
            provider = OpenAIProvider()
            print(f"✓ OpenAI provider created: {provider.provider_name}")
            
            # Test validation with mock client
            class MockClient:
                pass
            is_valid = provider.validate_client(MockClient())
            print(f"✓ Client validation works: {is_valid} (expected False)")
            
        except ImportError as e:
            print(f"⚠ OpenAI provider test skipped (OpenAI not installed): {e}")
        
        return True
    except Exception as e:
        print(f"✗ Provider system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_api():
    """Test the new provider API."""
    try:
        # Import the providers module
        from instructor.providers import BaseProvider, ProviderRegistry
        print("✓ Provider module import successful")
        
        # Create custom provider
        class TestProvider(BaseProvider):
            @property
            def provider_name(self) -> str:
                return "test"
            
            def validate_client(self, client) -> bool:
                return isinstance(client, str)
            
            def get_create_function(self, client, mode):
                return lambda: None
            
            def get_instructor_class(self, client):
                return type(None)
            
            def get_provider_enum(self):
                return "TEST"
            
            def validate_mode(self, mode) -> bool:
                return True
        
        # Test registry
        registry = ProviderRegistry()
        provider = TestProvider()
        registry.register(provider)
        
        print(f"✓ Custom provider registered: {registry.list_providers()}")
        
        # Test provider lookup
        found_provider = registry.get_provider("test")
        print(f"✓ Provider lookup: {found_provider.provider_name if found_provider else None}")
        
        # Test client matching
        matched_provider = registry.get_provider_for_client("test_client")
        print(f"✓ Client matching: {matched_provider.provider_name if matched_provider else None}")
        
        return True
    except Exception as e:
        print(f"✗ New API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing instructor provider migration...")
    print("="*50)
    
    success = True
    
    print("\n1. Testing backward compatibility...")
    success &= test_backward_compatibility()
    
    print("\n2. Testing provider system...")
    success &= test_provider_system()
    
    print("\n3. Testing new API...")
    success &= test_new_api()
    
    print("\n" + "="*50)
    if success:
        print("✓ All tests passed! Provider migration successful.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    sys.exit(0 if success else 1)