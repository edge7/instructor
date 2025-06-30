# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Unified asynchronous client parameter across the entire API.  Use `async_client: bool` going forward.

### Deprecated
- The parameters `_async` and `use_async` are now deprecated.  They continue to work for
  backwards-compatibility, but will raise a `DeprecationWarning`.  They will be
  removed in a future major release.

### Changed
- Internal wrappers (`from_vertexai`, `from_mistral`, `from_genai`, `from_gemini`,
  `from_bedrock`) now accept `async_client` instead of the legacy names.
- `instructor.auto_client.from_provider` now forwards the `async_client` flag to
  provider helpers.

### Documentation
- Updated all user-facing documentation, guides, and examples to use
  `async_client`.

---

Past changes can be found in the project's Git history.