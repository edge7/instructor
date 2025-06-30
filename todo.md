# Async Parameter Standardization TODO

## Completed
- [x] Updated `client_vertexai.py` to support `async_client` parameter with backwards compatibility
- [x] Updated `client_genai.py` to support `async_client` parameter with backwards compatibility  
- [x] Updated `client_mistral.py` to support `async_client` parameter with backwards compatibility
- [x] Updated `client_gemini.py` to support `async_client` parameter with backwards compatibility
- [x] Updated `client_bedrock.py` to support `async_client` parameter with backwards compatibility
- [x] Updated `auto_client.py` to use `async_client` when calling specific client functions
- [x] Updated all test files to use `async_client` instead of deprecated parameters
- [x] Created deprecation warning tests for all client libraries
- [x] Updated documentation files to use `async_client`

## Deprecation Strategy
- `_async` parameter (used in vertexai and bedrock) - shows deprecation warning, recommends `async_client`
- `use_async` parameter (used in genai, mistral, gemini) - shows deprecation warning, recommends `async_client`
- All parameters still work for backwards compatibility
- Error raised if multiple async parameters are provided

## Notes
- The `from_provider` API already uses `async_client` consistently
- Examples in the codebase already use `async_client` where applicable
- Linter errors in test files are expected (import resolution in test environment)