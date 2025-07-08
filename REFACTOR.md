# Provider System Refactor

## Core Implementation Principles
These principles establish core patterns that MUST be followed by all provider implementations throughout the refactor:

### Dependency Management
Provider implementations must handle dependencies by:
- Using importlib.util.find_spec() for conditional imports
- Implementing try/except blocks for provider-specific imports
- Providing helpful error messages with install instructions
- Using the `requires_package` decorator with proper version constraints
- Handling imports lazily to avoid unnecessary dependencies

Reference implementations:
- `instructor/providers/base/dependencies.py` - Core dependency utilities
- `instructor/cache/__init__.py` - Example of lazy imports
- `instructor/auto_client.py` - Current conditional imports
- `instructor/auto_client.py` - Provider-specific imports

### Type System
Provider type handling must:
- Implement type-related validation methods
- Re-use existing type utilities from dsl/simple_type.py
- Focus on create method type handling
- Maintain existing validation patterns

Reference implementations:
- `instructor/dsl/simple_type.py` - Type utilities
- `instructor/function_calls.py` - Core schema implementation
- `docs/concepts/types.md` - Type system documentation

### Retry System
Provider retry integration must:
- Implement retry configuration methods
- Define provider-specific retry conditions
- Maintain tenacity integration
- Preserve usage tracking and error context

Reference implementations:
- `instructor/retry.py` - Core retry implementation
- `instructor/patch.py` - Provider patching with retry support
- `examples/retry/run.py` - Example retry patterns

### Streaming Support
Provider streaming support must:
- Implement streaming-related abstract methods
- Re-use dsl/iterable.py for list streaming
- Re-use dsl/partial.py for partial streaming
- Add provider-specific stream processing

Reference implementations:
- `instructor/dsl/iterable.py` - List streaming implementation
- `instructor/dsl/partial.py` - Partial streaming implementation
- `docs/concepts/streaming.md` - Streaming documentation

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
  - [x] Add retry configuration:
    - [x] `_retry_config: dict[str, Any]` for tenacity settings
    - [x] Support for max_retries, timeout, and conditions
    - [x] Integration with create method retry logic
  - [x] Add create methods:
    - [x] `create(messages, response_model, **kwargs)` - Basic model creation
    - [x] `create_with_completion(messages, response_model, **kwargs)` - Returns (model, raw_response)
    - [x] `create_partial(response_model, messages, **kwargs)` - For streaming partial results
    - [x] `create_iterable(messages, response_model, **kwargs)` - For streaming list items
    ✅ Done: All create methods implemented with retry and hooks support
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

## Phase 2: OpenAI Provider - Core Implementation ✅
- [x] Create OpenAIProvider class
  - [x] Move mode handlers from process_response.py:
    - [x] `handle_functions`
    - [x] `handle_tools`
    - [x] `handle_tools_strict`
    - [x] `handle_json_modes`
  - [x] Move response processing from function_calls.py
  - [x] Add error handling from reask.py
  - [x] Verify implementation follows core principles
  - [x] Fix linting issues:
    - [x] Remove unused imports (cast, Set)
    - [x] Update to use built-in set type
  - [x] Add dependency management:
    - [x] Create base/dependencies.py with requires_package decorator
    - [x] Add OpenAI dependency check with min_version="1.0.0"
  - [x] Implement create methods:
    - [x] `create` method using OpenAI chat completions
    - [x] `create_with_completion` returning raw completion
    - [x] `create_partial` for streaming partial results
    - [x] `create_iterable` for streaming list items
    - [x] Integrate tenacity retry logic
    - [x] Handle validation errors and retries
  Files:
  - `instructor/providers/openai/__init__.py` - New OpenAI provider ✅
  - `instructor/providers/openai/response.py` - Response processing ✅
  - `instructor/providers/openai/errors.py` - Error handling ✅
  - `instructor/providers/base/dependencies.py` - Dependency utilities ✅
  Look at:
  - `instructor/process_response.py` - Current mode handlers
  - `instructor/function_calls.py` - Core schema implementation
  - `instructor/reask.py` - Current error handling
  - `instructor/dsl/partial.py` - Partial streaming implementation
  - `instructor/dsl/iterable.py` - List streaming implementation
  ✅ Done: Complete implementation with all features

## Phase 3: OpenAI Provider - Streaming
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

## Phase 4: OpenAI Provider - Factory Integration
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

## Phase 5A: Anthropic Provider - Core Implementation
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

## Phase 5B: Anthropic Provider - Error Handling
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

## Phase 5C: Anthropic Provider - Streaming
- [ ] Implement streaming support:
  - [ ] Re-use OpenAI streaming patterns where possible
  - [ ] Add Anthropic-specific streaming handlers
  Files:
  - `instructor/providers/anthropic/streaming.py` - Streaming implementation

## Phase 6: Cohere Provider
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

## Phase 7A: Google Provider - Core Implementation
- [ ] Move multimodal content handling
  Files:
  - `instructor/providers/google/__init__.py` - New Google provider
  - `instructor/providers/google/multimodal.py` - Content handling
  Look at:
  - `instructor/multimodal.py` - Current implementation
  - `tests/llm/test_gemini/` - Google-specific tests
  Current: Scattered across process_response.py
  Future: Encapsulated in GoogleProvider

## Phase 7B: Google Provider - Advanced Features
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

## Phase 8A: Mistral Provider
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

## Phase 8B: Bedrock Provider
- [ ] Bedrock
  - [ ] System message format
  - [ ] Response structure
  - [ ] Streaming support
  Files:
  - `instructor/providers/bedrock/` - New Bedrock provider

## Phase 9: Documentation Updates
- [ ] Add docstring examples to all providers
- [ ] Update API reference docs
- [ ] Add migration guide for provider system