# type: ignore
from __future__ import annotations

import re
from typing import Any, Literal, TypeVar, Union, overload

import jsonref
from google.genai import Client
from pydantic import BaseModel

import instructor
from instructor.multimodal import Audio, Image, PDF

T = TypeVar("T")


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_genai(
    client: Client,
    mode: instructor.Mode = instructor.Mode.GENAI_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.GENAI_TOOLS,
        instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="GenAI", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, Client):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of google.genai.Client. "
            f"Got: {type(client).__name__}"
        )

    if use_async:

        async def async_wrapper(*args: Any, **kwargs: Any):  # type:ignore
            if kwargs.pop("stream", False):
                return await client.aio.models.generate_content_stream(
                    *args, **kwargs
                )  # type:ignore
            return await client.aio.models.generate_content(
                *args, **kwargs
            )  # type:ignore

        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.GENAI,
            mode=mode,
            **kwargs,
        )

    def sync_wrapper(*args: Any, **kwargs: Any):  # type:ignore
        if kwargs.pop("stream", False):
            return client.models.generate_content_stream(*args, **kwargs)  # type:ignore

        return client.models.generate_content(*args, **kwargs)  # type:ignore

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=sync_wrapper, mode=mode),
        provider=instructor.Provider.GENAI,
        mode=mode,
        **kwargs,
    )


def handle_genai_structured_outputs(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    from google.genai import types

    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    # We validate that the schema doesn't contain any Union fields
    map_to_gemini_function_schema(response_model.model_json_schema())

    base_config = {
        "system_instruction": system_message,
        "response_mime_type": "application/json",
        "response_schema": response_model,
    }

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)

    return response_model, new_kwargs


def handle_genai_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    from google.genai import types

    schema = map_to_gemini_function_schema(response_model.model_json_schema())
    function_definition = types.FunctionDeclaration(
        name=response_model.__name__,
        description=response_model.__doc__,
        parameters=schema,
    )

    # We support the system message if you declare a system kwarg or if you pass a system message in the messages
    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    base_config = {
        "system_instruction": system_message,
        "tools": [types.Tool(function_declarations=[function_definition])],
        "tool_config": types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[response_model.__name__]
            ),
        ),
    }

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)

    return response_model, new_kwargs


# Mode handlers dictionary for GenAI
mode_handlers = {
    instructor.Mode.GENAI_TOOLS: handle_genai_tools,
    instructor.Mode.GENAI_STRUCTURED_OUTPUTS: handle_genai_structured_outputs,
}


def verify_no_unions(obj: dict[str, Any]) -> bool:
    """
    Verify that the object does not contain any Union types (except Optional).
    Optional[T] is allowed as it becomes Union[T, None].
    """
    for prop_value in obj["properties"].values():
        if "anyOf" in prop_value:
            # Check if this is an Optional type (Union with None/null)
            any_of_list = prop_value["anyOf"]
            if not isinstance(any_of_list, list) or len(any_of_list) != 2:
                return False

            # Check if one of the types is null (representing None in Optional[T])
            has_null = any(
                isinstance(item, dict) and item.get("type") == "null"
                for item in any_of_list
            )

            if not has_null:
                # This is a true Union type, not Optional - reject it
                return False

        if "properties" in prop_value and not verify_no_unions(prop_value):
            return False

    return True


def map_to_gemini_function_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Map OpenAPI schema to Gemini function call schema.

    Transforms a standard JSON schema to Gemini's expected format:
    - Adds 'format': 'enum' for enum fields
    - Converts Optional[T] (anyOf with null) to nullable fields
    - Rejects true Union types (non-Optional anyOf)

    Ref: https://ai.google.dev/api/python/google/generativeai/protos/Schema
    """

    class FunctionSchema(BaseModel):
        description: str | None = None
        enum: list[str] | None = None
        example: Any | None = None
        format: str | None = None
        nullable: bool | None = None
        items: FunctionSchema | None = None
        required: list[str] | None = None
        type: str
        properties: dict[str, FunctionSchema] | None = None

    # Resolve any $ref references in the schema
    schema: dict[str, Any] = jsonref.replace_refs(obj, lazy_load=False)  # type: ignore
    schema.pop("$defs", None)

    def transform_schema_node(node: Any) -> Any:
        """Transform a single schema node recursively."""
        if isinstance(node, list):
            return [transform_schema_node(item) for item in node]

        if not isinstance(node, dict):
            return node

        transformed = {}

        for key, value in node.items():
            if key == "enum":
                # Gemini requires 'format': 'enum' for enum fields
                transformed[key] = value
                transformed["format"] = "enum"
            elif key == "anyOf" and isinstance(value, list) and len(value) == 2:
                # Handle Optional[T] which becomes Union[T, None] in JSON schema
                non_null_items = [
                    item
                    for item in value
                    if not (isinstance(item, dict) and item.get("type") == "null")
                ]

                if len(non_null_items) == 1:
                    # This is Optional[T] - merge the actual type and mark as nullable
                    actual_type = transform_schema_node(non_null_items[0])
                    transformed.update(actual_type)
                    transformed["nullable"] = True
                else:
                    # This is a true Union type - keep as is and let validation catch it
                    transformed[key] = transform_schema_node(value)
            else:
                transformed[key] = transform_schema_node(value)

        return transformed

    schema = transform_schema_node(schema)

    # Validate that no unsupported Union types remain
    if not verify_no_unions(schema):
        raise ValueError(
            "Gemini does not support Union types (except Optional). Please change your function schema"
        )

    return FunctionSchema(**schema).model_dump(exclude_none=True, exclude_unset=True)


def update_genai_kwargs(
    kwargs: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Update keyword arguments for google.genai package from OpenAI format.
    """
    from google.genai.types import HarmCategory, HarmBlockThreshold

    new_kwargs = kwargs.copy()

    OPENAI_TO_GEMINI_MAP = {
        "max_tokens": "max_output_tokens",
        "temperature": "temperature",
        "n": "candidate_count",
        "top_p": "top_p",
        "stop": "stop_sequences",
        "seed": "seed",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
    }

    generation_config = new_kwargs.pop("generation_config", {})

    for openai_key, gemini_key in OPENAI_TO_GEMINI_MAP.items():
        if openai_key in generation_config:
            val = generation_config.pop(openai_key)
            if val is not None:  # Only set if value is not None
                base_config[gemini_key] = val

    safety_settings = new_kwargs.pop("safety_settings", {})
    base_config["safety_settings"] = []

    # Filter out image related harm categories which are not
    # supported for text based models
    supported_categories = [
        c
        for c in HarmCategory
        if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED
        and not c.name.startswith("HARM_CATEGORY_IMAGE_")
    ]

    for category in supported_categories:
        threshold = safety_settings.get(category, HarmBlockThreshold.OFF)
        base_config["safety_settings"].append(
            {
                "category": category,
                "threshold": threshold,
            }
        )

    return base_config


def extract_genai_system_message(
    messages: list[dict[str, Any]],
) -> str:
    """
    Extract system messages from a list of messages.

    We expect an explicit system messsage for this provider.
    """
    system_messages = ""

    for message in messages:
        if isinstance(message, str):
            continue
        elif isinstance(message, dict):
            if message.get("role") == "system":
                if isinstance(message.get("content"), str):
                    system_messages += message.get("content", "") + "\n\n"
                elif isinstance(message.get("content"), list):
                    for item in message.get("content", []):
                        if isinstance(item, str):
                            system_messages += item + "\n\n"

    if system_messages and len(messages) == 1:
        raise ValueError(
            "At least one user message must be included. A system message alone is not sufficient."
        )

    if re.search(r"{{.*?}}|{%.*?%}", system_messages):
        raise ValueError(
            "Jinja templating is not supported in system messages with Google GenAI, only user messages."
        )

    return system_messages


def convert_to_genai_messages(
    messages: list[Union[str, dict[str, Any], list[dict[str, Any]]]],  # noqa: UP007
) -> list[Any]:
    """
    Convert a list of messages to a list of dictionaries in the format expected by the Gemini API.

    This optimized version pre-allocates the result list and
    reduces function call overhead.
    """
    from google.genai import types

    result: list[Union[types.Content, types.File]] = []  # noqa: UP007

    for message in messages:
        # We assume this is the user's message and we don't need to convert it
        if isinstance(message, str):
            result.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            )
        elif isinstance(message, types.Content):
            result.append(message)
        elif isinstance(message, types.File):
            result.append(message)
        elif isinstance(message, dict):
            assert "role" in message
            assert "content" in message

            if message["role"] == "system":
                continue

            if message["role"] not in {"user", "model"}:
                raise ValueError(f"Unsupported role: {message['role']}")

            if isinstance(message["content"], str):
                result.append(
                    types.Content(
                        role=message["role"],
                        parts=[types.Part.from_text(text=message["content"])],
                    )
                )

            elif isinstance(message["content"], list):
                content_parts = []

                for content_item in message["content"]:
                    if isinstance(content_item, str):
                        content_parts.append(types.Part.from_text(text=content_item))
                    elif isinstance(content_item, (Image, Audio, PDF)):
                        content_parts.append(content_item.to_genai())
                    else:
                        raise ValueError(
                            f"Unsupported content item type: {type(content_item)}"
                        )

                result.append(
                    types.Content(
                        role=message["role"],
                        parts=content_parts,
                    )
                )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return result
