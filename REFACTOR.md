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

## Phase 2: OpenAI Provider Migration

- [ ] Create OpenAIProvider class
  - [ ] Move mode handlers from process_response.py:
    - [ ] `handle_functions`
    - [ ] `handle_tools`
    - [ ] `handle_tools_strict`
    - [ ] `handle_json_modes`
    Current: All handlers in process_response.py
    Future: Encapsulated in OpenAIProvider class
  - [ ] Move response processing from function_calls.py
    Current: OpenAISchema.from_response handles all formats
    Future: OpenAIProvider handles its specific formats
  - [ ] Add error handling from reask.py
    Current: Generic error handling with some OpenAI specifics
    Future: OpenAI-specific error handling in provider

- [ ] Update imports and factory functions
  - [ ] Update __init__.py to use provider registry
  - [ ] Maintain backwards compatibility for `from_openai`
  Current: Direct import of from_openai function
  Future: Factory function uses provider registry

## Phase 3: Provider Migrations

### Anthropic
- [ ] Move message format handling
  - [ ] System message extraction
    Current: handle_anthropic_tools extracts system messages
    Future: AnthropicProvider._extract_system_messages
  - [ ] Tool descriptions formatting
    Current: Uses anthropic_schema property
    Future: Provider handles schema conversion
  - [ ] JSON response parsing
    Current: parse_anthropic_json in OpenAISchema
    Future: AnthropicProvider._parse_json_response
- [ ] Move error handling
  - [ ] Validation error handling
  - [ ] Reask logic
  Current: reask_anthropic_json in reask.py
  Future: AnthropicProvider.handle_error

### Cohere
- [ ] Move message format handling
  - [ ] Convert to chat_history format
    Current: handle_cohere_modes converts messages
    Future: CohereProvider._prepare_chat_history
  - [ ] Handle role mappings
    Current: Hardcoded USER/CHATBOT mapping
    Future: Provider-defined role mapping
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

### Remaining Providers
- [ ] Mistral
  - [ ] Tools mode
  - [ ] Structured outputs
  Current: Mode-specific handlers in process_response.py
  Future: MistralProvider implements all modes

- [ ] Bedrock
  - [ ] System message format
  - [ ] Response structure
  Current: handle_bedrock_json and handle_bedrock_tools
  Future: BedrockProvider handles all formats

[... remaining providers follow same pattern ...]

## Phase 4: Testing Infrastructure

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
  Current: Mixed in with provider-specific tests
  Future: Common test suite for all providers

- [ ] Provider-specific tests
  - [ ] Message format tests
  - [ ] Response processing tests
  - [ ] Error handling tests
  - [ ] Mode support tests
  Current: Different test patterns per provider
  Future: Consistent test structure across providers

## Phase 5: Cleanup

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
  Current: Mixed documentation styles
  Future: Consistent provider documentation 