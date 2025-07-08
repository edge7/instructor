# Provider System Refactor

## Phase 1: Base Infrastructure ✅

- [x] Create directory structure
  ```bash
  mkdir -p instructor/providers/{base,registry,openai,anthropic,google,mistral,cohere}
  touch instructor/providers/__init__.py
  ```
  ✅ Done: Each provider gets its own directory with modular components

- [x] Implement BaseProvider
  - [x] Add abstract methods:
    - [x] `prepare_request(response_model, mode, **kwargs)`
      ✅ Done: Each provider implements this method for request preparation
    - [x] `process_response(response, response_model, mode, **kwargs)`
      ✅ Done: Each provider handles its own response format
    - [x] `handle_error(error, response, **kwargs)`
      ✅ Done: Each provider implements its error handling
  - [x] Add class attributes:
    - [x] `name: str`
    - [x] `supported_modes: ClassVar[set[Mode]]`
      ✅ Done: Each provider declares supported modes

- [x] Implement ProviderRegistry
  - [x] Add provider registration decorator
  - [x] Add provider lookup method
  - [x] Add provider validation
  ✅ Done: Centralized provider registration and lookup

### Additional Improvements Made
- Updated type hints to use modern Python syntax:
  - Replaced `typing.Dict` with `dict`
  - Replaced `typing.Type` with `type`
  - Replaced `typing.Set` with `set`

## Phase 2: Base Provider Enhancements

- [ ] Add retry support to BaseProvider
  - [ ] Move retry logic from instructor/retry.py
  - [ ] Add retry-related abstract methods:
    - [ ] `initialize_retry_config(max_retries, timeout)`
    - [ ] `handle_retry_error(error, response, attempt)`
  - [ ] Re-use existing tenacity integration:
    - [ ] Support sync/async retrying
    - [ ] Keep existing retry conditions
    - [ ] Preserve usage tracking
  Current: Retry logic in retry.py
  Future: Provider-specific retry handling

- [ ] Add type handling to BaseProvider
  - [ ] Move type utilities from dsl/simple_type.py
  - [ ] Add type-related methods:
    - [ ] `prepare_response_model(response_model)`
    - [ ] `validate_response_type(response, response_model)`
  - [ ] Re-use existing type validation:
    - [ ] Simple type wrapping
    - [ ] Iterable type handling
    - [ ] Union type support
  Current: Type handling scattered across codebase
  Future: Centralized in BaseProvider with provider-specific overrides

- [ ] Add streaming support to BaseProvider
  - [ ] Re-use existing streaming implementations:
    - [ ] Iterable streaming from dsl/iterable.py
    - [ ] Partial streaming from dsl/partial.py
  - [ ] Add streaming-related abstract methods:
    - [ ] `process_streaming_response_async(response, response_model, mode, **kwargs)`
    - [ ] `process_streaming_response(response, response_model, mode, **kwargs)`
  - [ ] Add streaming-specific type hints and validation
  Current: Streaming logic in dsl/ modules
  Future: Provider-specific streaming implementations

## Phase 3: OpenAI Provider Migration

- [ ] Create OpenAIProvider class
  - [ ] Move mode handlers from process_response.py:
    - [ ] `handle_functions`
    - [ ] `handle_tools`
    - [ ] `handle_tools_strict`
    - [ ] `handle_json_modes`
  - [ ] Move response processing from function_calls.py
  - [ ] Add error handling from reask.py
  - [ ] Implement streaming methods:
    - [ ] Re-use IterableBase for list streaming
    - [ ] Re-use PartialBase for partial streaming
  Current: All handlers in process_response.py
  Future: Encapsulated in OpenAIProvider class

- [ ] Update imports and factory functions
  - [ ] Update __init__.py to use provider registry
  - [ ] Maintain backwards compatibility for `from_openai`
  Current: Direct import of from_openai function
  Future: Factory function uses provider registry

## Phase 4: Provider Migrations

### Anthropic
- [ ] Move message format handling
  - [ ] System message extraction
  - [ ] Tool descriptions formatting
  - [ ] JSON response parsing
  - [ ] Implement streaming support:
    - [ ] Re-use OpenAI streaming patterns where possible
    - [ ] Add Anthropic-specific streaming handlers
  Current: Mixed handlers across files
  Future: Encapsulated in AnthropicProvider

- [ ] Move error handling
  - [ ] Validation error handling
  - [ ] Reask logic
  Current: reask_anthropic_json in reask.py
  Future: AnthropicProvider.handle_error

### Cohere
- [ ] Move message format handling
  - [ ] Convert to chat_history format
  - [ ] Handle role mappings
  - [ ] Implement streaming methods
  Current: Mixed handlers
  Future: CohereProvider implementation

- [ ] Move response processing
  Current: parse_cohere_tools in OpenAISchema
  Future: CohereProvider._parse_response

### Google/Vertex
- [ ] Move multimodal content handling
  Current: Scattered across process_response.py
  Future: Encapsulated in GoogleProvider

- [ ] Move parallel processing support
  Current: handle_vertexai_parallel_tools
  Future: GoogleProvider._handle_parallel

- [ ] Add streaming support
  - [ ] Implement Google-specific streaming
  - [ ] Handle async streaming properly
  Current: Basic streaming support
  Future: Full streaming capabilities

### Remaining Providers
- [ ] Mistral
  - [ ] Tools mode
  - [ ] Structured outputs
  - [ ] Streaming implementation
  Current: Mode-specific handlers
  Future: MistralProvider implements all modes

- [ ] Bedrock
  - [ ] System message format
  - [ ] Response structure
  - [ ] Streaming support
  Current: Basic handlers
  Future: BedrockProvider handles all formats

[... remaining providers follow same pattern ...]

## Phase 5: Testing Infrastructure

- [ ] Create test structure
  ```bash
  mkdir -p tests/providers/{base,openai,anthropic,google,mistral,cohere}
  touch tests/providers/conftest.py
  ```
  Current: Tests in tests/llm/test_*/
  Future: Unified provider test structure

- [ ] Base provider tests
  - [ ] Registration tests
  - [ ] Mode validation tests
  - [ ] Error handling tests
  - [ ] Streaming tests:
    - [ ] Sync streaming
    - [ ] Async streaming
    - [ ] Type validation during streaming
  Current: Mixed test coverage
  Future: Comprehensive test suite

- [ ] Provider-specific tests
  - [ ] Message format tests
  - [ ] Response processing tests
  - [ ] Error handling tests
  - [ ] Mode support tests
  - [ ] Streaming behavior tests:
    - [ ] List streaming
    - [ ] Partial streaming
    - [ ] Error handling during streaming
  Current: Different test patterns
  Future: Consistent test structure

## Phase 6: Cleanup and Documentation

- [ ] Remove deprecated files
  ```bash
  rm instructor/client_*.py
  ```
  Current: Provider logic in client_*.py files
  Future: All providers in providers/ directory

- [ ] Update documentation
  - [ ] Update docstrings
  - [ ] Update type hints
  - [ ] Update examples
  - [ ] Add streaming documentation:
    - [ ] Provider-specific streaming capabilities
    - [ ] Sync vs async streaming patterns
    - [ ] Type handling during streaming
  - [ ] Add async/sync usage documentation
  Current: Mixed documentation styles
  Future: Consistent provider documentation 