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

## Phase 2A: Dependency Management
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
  - `instructor/client.py` - Base client implementation
  Current: Mixed dependency handling in __init__.py and auto_client.py
  Future: Consistent dependency handling across all providers

## Phase 2B: Retry System Integration
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
  - `instructor/process_response.py` - Response processing with retries
  - `instructor/client.py` - Client retry configuration
  - `docs/concepts/retrying.md` - Retry documentation
  - `examples/tenacity-benchmarks/run.py` - Example retry patterns
  - `examples/retry/run.py` - Additional retry examples
  Current: Retry logic in retry.py, used by patch.py
  Future: Provider-specific retry configuration with shared core implementation

## Phase 2C: Type System Integration
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
  - `instructor/function_calls.py` - Core OpenAISchema implementation
  - `instructor/process_response.py` - Response processing
  Look at:
  - `instructor/function_calls.py` - Current OpenAISchema (core schema spec)
  - `instructor/dsl/simple_type.py` - Current type validation
  - `tests/dsl/test_simple_type.py` - Type validation tests
  - `docs/concepts/types.md` - Type system documentation
  - `docs/learning/validation/field_level_validation.md` - Field validation patterns
  Current: Type handling in dsl/ and process_response.py
  Future: Provider-specific type validation with shared utilities

## Phase 2D: Streaming Support
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
  - `instructor/function_calls.py` - Core schema streaming support
  Look at:
  - `instructor/dsl/iterable.py` - Current list streaming
  - `instructor/dsl/partial.py` - Current partial streaming
  - `instructor/function_calls.py` - Current streaming schema handling
  - `docs/concepts/streaming.md` - Streaming documentation
  Current: Streaming logic in dsl/ modules
  Future: Provider-specific streaming with shared core implementations

## Phase 3A: OpenAI Provider - Core Implementation
- [ ] Create OpenAIProvider class
  - [ ] Move mode handlers from process_response.py:
    - [ ] `handle_functions`
    - [ ] `handle_tools`
    - [ ] `handle_tools_strict`
    - [ ] `handle_json_modes`
  - [ ] Move response processing from function_calls.py
  - [ ] Add error handling from reask.py
  Files:
  - `instructor/providers/openai/__init__.py` - New OpenAI provider
  - `instructor/providers/openai/response.py` - Response processing
  - `instructor/providers/openai/errors.py` - Error handling

## Phase 3B: OpenAI Provider - Streaming
- [ ] Implement streaming methods:
  - [ ] Re-use IterableBase for list streaming
  - [ ] Re-use PartialBase for partial streaming
  Files:
  - `instructor/providers/openai/streaming.py` - Streaming implementation
  Look at:
  - `instructor/process_response.py` - Current mode handlers
  - `instructor/function_calls.py` - Core schema implementation
  - `instructor/reask.py` - Current error handling
  - `instructor/dsl/iterable.py` - List streaming base
  - `instructor/dsl/partial.py` - Partial streaming base
  - `instructor/client.py` - Current OpenAI client implementation
  - `instructor/patch.py` - Current OpenAI patching
  - `tests/llm/test_openai/` - OpenAI-specific tests
  Current: All handlers in process_response.py
  Future: Encapsulated in OpenAIProvider class

## Phase 3C: OpenAI Provider - Factory Integration
- [ ] Update imports and factory functions
  - [ ] Update __init__.py to use provider registry
  - [ ] Maintain backwards compatibility for `from_openai`
  Files:
  - `instructor/__init__.py` - Updated imports
  - `instructor/auto_client.py` - Updated factory functions
  Look at:
  - `instructor/client.py` - Current client initialization
  - `instructor/auto_client.py` - Current factory pattern
  - `tests/test_auto_client.py` - Factory function tests
  Current: Direct import of from_openai function
  Future: Factory function uses provider registry

## Phase 4A: Anthropic Provider - Core Implementation
- [ ] Move message format handling
  - [ ] System message extraction
  - [ ] Tool descriptions formatting
  - [ ] JSON response parsing
  Files:
  - `instructor/providers/anthropic/__init__.py` - New Anthropic provider
  - `instructor/providers/anthropic/messages.py` - Message handling
  - `instructor/providers/anthropic/tools.py` - Tool handling
  Look at:
  - `instructor/client_anthropic.py` - Current implementation
  - `tests/llm/test_anthropic/` - Anthropic-specific tests
  Current: Mixed handlers across files
  Future: Encapsulated in AnthropicProvider

## Phase 4B: Anthropic Provider - Error Handling
- [ ] Move error handling
  - [ ] Validation error handling
  - [ ] Reask logic
  Files:
  - `instructor/providers/anthropic/errors.py` - Error handling
  Look at:
  - `instructor/reask.py` - Current reask logic
  - `tests/llm/test_anthropic/evals/` - Error handling tests
  Current: reask_anthropic_json in reask.py
  Future: AnthropicProvider.handle_error

## Phase 4C: Anthropic Provider - Streaming
- [ ] Implement streaming support:
  - [ ] Re-use OpenAI streaming patterns where possible
  - [ ] Add Anthropic-specific streaming handlers
  Files:
  - `instructor/providers/anthropic/streaming.py` - Streaming implementation

## Phase 5: Cohere Provider
- [ ] Move message format handling
  - [ ] Convert to chat_history format
  - [ ] Handle role mappings
  - [ ] Implement streaming methods
  Files:
  - `instructor/providers/cohere/__init__.py` - New Cohere provider
  - `instructor/providers/cohere/chat.py` - Chat handling
  Look at:
  - `instructor/client_cohere.py` - Current implementation
  - `tests/llm/test_cohere/` - Cohere-specific tests
  Current: Mixed handlers
  Future: CohereProvider implementation

- [ ] Move response processing
  Files:
  - `instructor/providers/cohere/response.py` - Response processing
  Look at:
  - `instructor/process_response.py` - Current processing
  Current: parse_cohere_tools in OpenAISchema
  Future: CohereProvider._parse_response

## Phase 6A: Google Provider - Core Implementation
- [ ] Move multimodal content handling
  Files:
  - `instructor/providers/google/__init__.py` - New Google provider
  - `instructor/providers/google/multimodal.py` - Content handling
  Look at:
  - `instructor/multimodal.py` - Current implementation
  - `tests/llm/test_gemini/` - Google-specific tests
  Current: Scattered across process_response.py
  Future: Encapsulated in GoogleProvider

## Phase 6B: Google Provider - Advanced Features
- [ ] Move parallel processing support
  Files:
  - `instructor/providers/google/parallel.py` - Parallel processing
  Look at:
  - `instructor/dsl/parallel.py` - Current implementation
  Current: handle_vertexai_parallel_tools
  Future: GoogleProvider._handle_parallel

- [ ] Add streaming support
  - [ ] Implement Google-specific streaming
  - [ ] Handle async streaming properly
  Files:
  - `instructor/providers/google/streaming.py` - Streaming support
  Look at:
  - `instructor/dsl/iterable.py` - Current streaming base
  Current: Basic streaming support
  Future: Full streaming capabilities

## Phase 7A: Mistral Provider
- [ ] Mistral
  - [ ] Tools mode
  - [ ] Structured outputs
  - [ ] Streaming implementation
  Files:
  - `instructor/providers/mistral/` - New Mistral provider
  Look at:
  - `instructor/client_mistral.py` - Current implementation
  - `tests/llm/test_mistral/` - Mistral-specific tests
  Current: Mode-specific handlers
  Future: MistralProvider implements all modes

## Phase 7B: Bedrock Provider
- [ ] Bedrock
  - [ ] System message format
  - [ ] Response structure
  - [ ] Streaming support
  Files:
  - `instructor/providers/bedrock/` - New Bedrock provider