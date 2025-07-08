# Provider System Refactor

## Phase 1: Base Infrastructure ✅

- [x] Create directory structure
  ```bash
  mkdir -p instructor/providers/{base,registry,openai,anthropic,google,mistral,cohere}
  touch instructor/providers/__init__.py
  ```
  Look at:
  - `instructor/` - Current module structure
  - `instructor/providers/` - New provider directory
  - `AGENT.md` - Core library architecture
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
  Look at:
  - `instructor/client.py` - Current base client implementation
  - `instructor/client_*.py` - Current provider-specific clients
  - `instructor/process_response.py` - Response processing logic
  - `instructor/mode.py` - Mode system implementation
  - `tests/llm/test_*/` - Provider-specific tests

- [x] Implement ProviderRegistry
  - [x] Add provider registration decorator
  - [x] Add provider lookup method
  - [x] Add provider validation
  Look at:
  - `instructor/auto_client.py` - Current provider factory pattern
  - `instructor/utils.py` - Current provider detection
  - `instructor/providers/registry/__init__.py` - New registry implementation
  - `tests/test_auto_client.py` - Provider factory tests
  ✅ Done: Centralized provider registration and lookup

### Additional Improvements Made
- Updated type hints to use modern Python syntax:
  - Replaced `typing.Dict` with `dict`
  - Replaced `typing.Type` with `type`
  - Replaced `typing.Set` with `set`
  Look at:
  - `pyproject.toml` - Python version requirements
  - `instructor/function_calls.py` - Type usage examples
  - `instructor/dsl/simple_type.py` - Type handling utilities

## Phase 2: Base Provider Enhancements
These enhancements establish core patterns that must be followed by all future provider implementations:

- [ ] Re-use dependency handling patterns
  - [ ] Keep importlib.util.find_spec() for conditional imports
  - [ ] Keep try/except blocks for provider-specific imports
  - [ ] Keep helpful error messages with install instructions
  Files:
  - `instructor/providers/base/dependencies.py` - Core dependency utilities
  - `instructor/providers/base/__init__.py` - Base provider integration
  - `instructor/providers/*/provider.py` - Provider-specific implementations
  Look at:
  - `instructor/__init__.py` - Current conditional imports
  - `instructor/auto_client.py` - Current provider-specific imports
  - `instructor/cache/__init__.py` - Example of lazy imports
  Current: Mixed dependency handling in __init__.py and auto_client.py
  Future: Consistent dependency handling across all providers

- [ ] Integrate retry system with BaseProvider
  - [ ] Add retry configuration methods:
    - [ ] `configure_retry(max_retries, timeout)` - Configure provider-specific retry settings
    - [ ] `get_retry_conditions()` - Get provider-specific retry conditions
  - [ ] Keep core retry.py implementation:
    - [ ] Used by patch.py for all providers
    - [ ] Maintains tenacity integration
    - [ ] Preserves usage tracking
    - [ ] Keeps error context
  Files:
  - `instructor/providers/base/__init__.py` - Base retry configuration
  - `instructor/retry.py` - Core retry implementation
  - `instructor/patch.py` - Provider patching with retry support
  Look at:
  - `instructor/retry.py` - Current retry implementation
  - `instructor/patch.py` - Current patching system
  - `docs/concepts/retrying.md` - Retry documentation
  - `examples/tenacity-benchmarks/run.py` - Example retry patterns
  Current: Retry logic in retry.py, used by patch.py
  Future: Provider-specific retry configuration with shared core implementation

- [ ] Add type handling to BaseProvider
  - [ ] Add type-related abstract methods:
    - [ ] `validate_response_type(response, response_model)` - Provider-specific type validation
  - [ ] Re-use existing type utilities:
    - [ ] Keep dsl/simple_type.py implementation
    - [ ] Focus on create method type handling
    - [ ] Maintain existing validation patterns
  Files:
  - `instructor/providers/base/__init__.py` - Base type validation
  - `instructor/dsl/simple_type.py` - Type utilities
  - `instructor/process_response.py` - Response processing
  Look at:
  - `instructor/dsl/simple_type.py` - Current type validation
  - `tests/dsl/test_simple_type.py` - Type validation tests
  - `docs/concepts/types.md` - Type system documentation
  - `docs/learning/validation/field_level_validation.md` - Field validation patterns
  Current: Type handling in dsl/ and process_response.py
  Future: Provider-specific type validation with shared utilities

- [ ] Add streaming support to BaseProvider
  - [ ] Add streaming-related abstract methods:
    - [ ] `process_streaming_response(response, response_model, mode, **kwargs)`
    - [ ] `process_streaming_response_async(response, response_model, mode, **kwargs)`
  - [ ] Re-use existing streaming implementations:
    - [ ] Keep dsl/iterable.py for list streaming
    - [ ] Keep dsl/partial.py for partial streaming
    - [ ] Add provider-specific stream processing
  Files:
  - `instructor/providers/base/__init__.py` - Base streaming methods
  - `instructor/dsl/iterable.py` - List streaming implementation
  - `instructor/dsl/partial.py` - Partial streaming implementation
  Look at:
  - `instructor/dsl/iterable.py` - Current list streaming
  - `instructor/dsl/partial.py` - Current partial streaming
  - `docs/concepts/streaming.md` - Streaming documentation
  Current: Streaming logic in dsl/ modules
  Future: Provider-specific streaming with shared core implementations

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