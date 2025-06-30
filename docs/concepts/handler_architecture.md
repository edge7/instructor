# Handler Architecture (Experimental)

> **Status:** Experimental in vNEXT – currently covers Anthropic modes. More providers will migrate in future releases.

Instructor now uses a plug-in style *handler* system to prepare requests and parse responses.  
Every provider / mode combination can live in its own file, making the codebase easier to extend and maintain.

## Why a new layer?

* The former `handle_response_model` function ballooned beyond 1 000 lines.
* Adding a new provider meant appending yet another `elif` block.
* Testing isolated behaviour was hard because of tight coupling.

## Core interfaces

```python
from instructor.handlers.base import ResponseHandler
```

A concrete handler must implement:

* `supported_modes`: tuple of `Mode` values it is responsible for.
* `prepare_request(..) -> (response_model, kwargs)` – mutate the outgoing kwargs **without touching the original dict**.
* `parse_response(..) -> Any | None` – convert the raw provider completion into the final return type or return `None` to delegate to the legacy parser.

Handlers register themselves via:

```python
from instructor.handlers.registry import register

register(MyHandler())
```

## Current builtin handler

* `AnthropicHandler` – covers the following modes:
  * `Mode.ANTHROPIC_TOOLS`
  * `Mode.ANTHROPIC_REASONING_TOOLS`
  * `Mode.ANTHROPIC_JSON`

## Extending Instructor with your own handler

```python
from instructor.mode import Mode
from instructor.handlers.base import ResponseHandler
from instructor.handlers.registry import register

class MyCustomHandler(ResponseHandler):
    supported_modes = (Mode.GEMINI_TOOLS,)

    def prepare_request(self, mode, response_model, call_kwargs):
        # mutate call_kwargs here
        return response_model, call_kwargs

    def parse_response(self, mode, raw_completion, *, response_model, stream, validation_context, strict):
        # convert raw_completion here
        return parsed_object

register(MyCustomHandler())
```

## Deprecation timeline

The legacy monolithic path is still available but raises a `DeprecationWarning` when used.  
It will be removed once all major providers have migrated to the new handler system.

---

*Questions?* Join the discussion on GitHub or [Discord](https://discord.gg/instructor).