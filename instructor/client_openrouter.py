# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def handle_openrouter_structured_outputs(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": schema,
            "strict": True,
        },
    }
    return response_model, new_kwargs


# Need to import instructor for Mode enum without circular import
import instructor

# Mode handlers mapping
mode_handlers = {
    instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS: handle_openrouter_structured_outputs,
}
