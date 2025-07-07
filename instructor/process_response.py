# type: ignore[all]
from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Iterable
from textwrap import dedent
from typing import Any, TypeVar, get_args, get_origin

from openai import pydantic_function_tool
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, create_model
from typing_extensions import ParamSpec

from instructor.dsl.iterable import IterableBase, IterableModel
from instructor.dsl.parallel import (
    ParallelBase,
    ParallelModel,
    VertexAIParallelBase,
    handle_parallel_model,
)
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import (
    AdapterBase,
    ModelAdapter,
    is_simple_type,
)
from instructor.function_calls import OpenAISchema, openai_schema
from instructor.mode import Mode
from instructor.multimodal import convert_messages, extract_genai_multimodal_content
from instructor.utils import extract_system_messages, merge_consecutive_messages
import importlib.util

if importlib.util.find_spec("anthropic") is not None:
    from .client_anthropic import (
        handle_anthropic_json,
        handle_anthropic_reasoning_tools,
        handle_anthropic_tools,
    )

if importlib.util.find_spec("cohere") is not None:
    from .client_cohere import (
        handle_cohere_json_schema,
        handle_cohere_modes,
        handle_cohere_tools,
    )

if importlib.util.find_spec("mistralai") is not None:
    from .client_mistral import (
        handle_mistral_structured_outputs,
        handle_mistral_tools,
    )

if importlib.util.find_spec("fireworks") is not None:
    from .client_fireworks import handle_fireworks_json, handle_fireworks_tools

if importlib.util.find_spec("google.generativeai") is not None:
    from .client_gemini import handle_gemini_json, handle_gemini_tools

if importlib.util.find_spec("google.genai") is not None:
    from .client_genai import handle_genai_structured_outputs, handle_genai_tools

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    from .client_vertexai import (
        handle_vertexai_json,
        handle_vertexai_parallel_tools,
        handle_vertexai_tools,
    )

if importlib.util.find_spec("boto3") is not None:
    from .client_bedrock import handle_bedrock_json, handle_bedrock_tools

if importlib.util.find_spec("cerebras") is not None:
    from .client_cerebras import handle_cerebras_json, handle_cerebras_tools

if importlib.util.find_spec("writerai") is not None:
    from .client_writer import handle_writer_json, handle_writer_tools

if importlib.util.find_spec("openai") is not None:
    from .client_perplexity import handle_perplexity_json

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """
    Asynchronously processes the response from the OpenAI API.

    Args:
        response (ChatCompletion): The raw response from the OpenAI API.
        response_model (type[T_Model | OpenAISchema | BaseModel] | None): The expected model type for the response.
        stream (bool): Whether the response is streamed.
        validation_context (dict[str, Any] | None): Additional context for validation.
        strict (bool | None): Whether to apply strict validation.
        mode (Mode): The processing mode to use.

    Returns:
        T_Model | ChatCompletion: The processed response, either as the specified model type or the raw ChatCompletion.

    This function handles various response types, including streaming responses and different model bases.
    It applies the appropriate processing based on the response_model and mode provided.
    """

    logger.debug(
        f"Instructor Raw Response: {response}",
    )
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response
    return model


def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | VertexAIParallelBase | None:
    """
    Process the response from the API call and convert it to the specified response model.

    Args:
        response (T_Model): The raw response from the API call.
        response_model (type[OpenAISchema | BaseModel] | None): The model to convert the response to.
        stream (bool): Whether the response is a streaming response.
        validation_context (dict[str, Any] | None): Additional context for validation.
        strict (bool | None): Whether to use strict validation.
        mode (Mode): The mode used for processing the response.

    Returns:
        The processed response, which could be:
        - The raw response if no response_model is specified
        - An instance of the response_model
        - A list of tasks if the model is an IterableBase
        - The content of the model if it's an AdapterBase

    This function handles various types of responses and models, including streaming
    responses, iterable models, parallel models, and adapter models. It also attaches
    the raw response to the processed model when applicable.
    """
    logger.debug(
        f"Instructor Raw Response: {response}",
    )

    if response_model is None:
        logger.debug("No response model, returning response as is")
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        logger.debug(f"Returning takes from IterableBase")
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        logger.debug(f"Returning model from ParallelBase")
        return model

    if isinstance(model, AdapterBase):
        logger.debug(f"Returning model from AdapterBase")
        return model.content

    model._raw_response = response

    return model


def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def handle_parallel_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    if new_kwargs.get("stream", False):
        from instructor.exceptions import ConfigurationError

        raise ConfigurationError(
            "stream=True is not supported when using PARALLEL_TOOLS mode"
        )
    new_kwargs["tools"] = handle_parallel_model(response_model)
    new_kwargs["tool_choice"] = "auto"
    return ParallelModel(typehint=response_model), new_kwargs


def handle_functions(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    Mode.warn_mode_functions_deprecation()
    new_kwargs["functions"] = [response_model.openai_schema]
    new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}
    return response_model, new_kwargs


def handle_tools_strict(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    response_model_schema = pydantic_function_tool(response_model)
    response_model_schema["function"]["strict"] = True
    new_kwargs["tools"] = [response_model_schema]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model_schema["function"]["name"]},
    }
    return response_model, new_kwargs


def handle_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": response_model.openai_schema,
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model.openai_schema["name"]},
    }
    return response_model, new_kwargs


def handle_responses_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    schema = pydantic_function_tool(response_model)
    del schema["function"]["strict"]

    tool_definition = {
        "type": "function",
        "name": schema["function"]["name"],
        "parameters": schema["function"]["parameters"],
    }

    if "description" in schema["function"]:
        tool_definition["description"] = schema["function"]["description"]
    else:
        tool_definition["description"] = (
            f"Correctly extracted `{response_model.__name__}` with all "
            f"the required parameters with correct types"
        )

    new_kwargs["tools"] = [
        {
            "type": "function",
            "name": schema["function"]["name"],
            "parameters": schema["function"]["parameters"],
        }
    ]

    new_kwargs["tool_choice"] = {
        "type": "function",
        "name": response_model.openai_schema["name"],
    }
    if new_kwargs.get("max_tokens") is not None:
        new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

    return response_model, new_kwargs


def handle_responses_tools_with_inbuilt_tools(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    schema = pydantic_function_tool(response_model)
    del schema["function"]["strict"]

    tool_definition = {
        "type": "function",
        "name": schema["function"]["name"],
        "parameters": schema["function"]["parameters"],
    }

    if "description" in schema["function"]:
        tool_definition["description"] = schema["function"]["description"]
    else:
        tool_definition["description"] = (
            f"Correctly extracted `{response_model.__name__}` with all "
            f"the required parameters with correct types"
        )

    if not new_kwargs.get("tools"):
        new_kwargs["tools"] = [tool_definition]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "name": response_model.openai_schema["name"],
        }
    else:
        new_kwargs["tools"].append(tool_definition)

    if new_kwargs.get("max_tokens") is not None:
        new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

    return response_model, new_kwargs


def handle_json_o1(
    response_model: type[T], new_kwargs: dict[str, Any]
) -> tuple[type[T], dict[str, Any]]:
    roles = [message["role"] for message in new_kwargs.get("messages", [])]
    if "system" in roles:
        raise ValueError("System messages are not supported For the O1 models")

    message = dedent(
        f"""
        Understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    new_kwargs["messages"].append(
        {
            "role": "user",
            "content": message,
        },
    )
    return response_model, new_kwargs


def handle_json_modes(
    response_model: type[T], new_kwargs: dict[str, Any], mode: Mode
) -> tuple[type[T], dict[str, Any]]:
    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    if mode == Mode.JSON:
        new_kwargs["response_format"] = {"type": "json_object"}
    elif mode == Mode.JSON_SCHEMA:
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        }
    elif mode == Mode.MD_JSON:
        new_kwargs["messages"].append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        )
        new_kwargs["messages"] = merge_consecutive_messages(new_kwargs["messages"])

    if new_kwargs["messages"][0]["role"] != "system":
        new_kwargs["messages"].insert(
            0,
            {
                "role": "system",
                "content": message,
            },
        )
    elif isinstance(new_kwargs["messages"][0]["content"], str):
        new_kwargs["messages"][0]["content"] += f"\n\n{message}"
    elif isinstance(new_kwargs["messages"][0]["content"], list):
        new_kwargs["messages"][0]["content"][0]["text"] += f"\n\n{message}"
    else:
        raise ValueError(
            "Invalid message format, must be a string or a list of messages"
        )

    return response_model, new_kwargs


def prepare_response_model(response_model: type[T] | None) -> type[T] | None:
    """
    Prepares the response model for use in the API call.

    This function performs several transformations on the input response_model:
    1. If the response_model is None, it returns None.
    2. If it's a simple type, it wraps it in a ModelAdapter.
    3. If it's a TypedDict, it converts it to a Pydantic BaseModel.
    4. If it's an Iterable, it wraps the element type in an IterableModel.
    5. If it's not already a subclass of OpenAISchema, it applies the openai_schema decorator.

    Args:
        response_model (type[T] | None): The input response model to be prepared.

    Returns:
        type[T] | None: The prepared response model, or None if the input was None.
    """
    if response_model is None:
        return None

    if is_simple_type(response_model):
        response_model = ModelAdapter[response_model]

    if is_typed_dict(response_model):
        response_model: BaseModel = create_model(
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},
        )

    if get_origin(response_model) is Iterable:
        iterable_element_class = get_args(response_model)[0]
        response_model = IterableModel(iterable_element_class)

    if not issubclass(response_model, OpenAISchema):
        response_model = openai_schema(response_model)  # type: ignore

    return response_model


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


def handle_response_model(
    response_model: type[T] | None, mode: Mode = Mode.TOOLS, **kwargs: Any
) -> tuple[type[T] | VertexAIParallelBase | None, dict[str, Any]]:
    """
    Handles the response model based on the specified mode and prepares the kwargs for the API call.

    Args:
        response_model (type[T] | None): The response model to be used for parsing the API response.
        mode (Mode): The mode to use for handling the response model. Defaults to Mode.TOOLS.
        **kwargs: Additional keyword arguments to be passed to the API call.

    Returns:
        tuple[type[T] | None, dict[str, Any]]: A tuple containing the processed response model and the updated kwargs.

    This function prepares the response model and modifies the kwargs based on the specified mode.
    It handles various modes like TOOLS, JSON, FUNCTIONS, etc., and applies the appropriate
    transformations to the response model and kwargs.
    """

    new_kwargs = kwargs.copy()
    # print(f"instructor.process_response.py: new_kwargs -> {new_kwargs}")
    autodetect_images = new_kwargs.pop("autodetect_images", False)

    if response_model is None:
        if mode in {Mode.COHERE_JSON_SCHEMA, Mode.COHERE_TOOLS}:
            # This is cause cohere uses 'message' and 'chat_history' instead of 'messages'
            return handle_cohere_modes(new_kwargs)
        # Handle images without a response model
        if "messages" in new_kwargs:
            messages = convert_messages(
                new_kwargs["messages"],
                mode,
                autodetect_images=autodetect_images,
            )
            if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
                # Handle OpenAI style or Anthropic style messages
                new_kwargs["messages"] = [m for m in messages if m["role"] != "system"]
                if "system" not in new_kwargs:
                    system_message = extract_system_messages(messages)
                    if system_message:
                        new_kwargs["system"] = system_message

            else:
                if mode in {
                    Mode.RESPONSES_TOOLS,
                    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                } and new_kwargs.get("max_tokens"):
                    new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

                new_kwargs["messages"] = messages
        return None, new_kwargs

    if mode in {Mode.PARALLEL_TOOLS}:
        return handle_parallel_tools(response_model, new_kwargs)
    elif mode in {Mode.VERTEXAI_PARALLEL_TOOLS}:
        return handle_vertexai_parallel_tools(response_model, new_kwargs)

    response_model = prepare_response_model(response_model)

    mode_handlers = {  # type: ignore
        Mode.FUNCTIONS: handle_functions,
        Mode.TOOLS_STRICT: handle_tools_strict,
        Mode.TOOLS: handle_tools,
        Mode.MISTRAL_TOOLS: handle_mistral_tools,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: handle_mistral_structured_outputs,
        Mode.JSON_O1: handle_json_o1,
        Mode.JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON),  # type: ignore
        Mode.MD_JSON: lambda rm, nk: handle_json_modes(rm, nk, Mode.MD_JSON),  # type: ignore
        Mode.JSON_SCHEMA: lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON_SCHEMA),  # type: ignore
        Mode.OPENROUTER_STRUCTURED_OUTPUTS: handle_openrouter_structured_outputs,
        Mode.RESPONSES_TOOLS: handle_responses_tools,
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: handle_responses_tools_with_inbuilt_tools,
    }

    if importlib.util.find_spec("anthropic") is not None:
        mode_handlers.update(
            {
                Mode.ANTHROPIC_TOOLS: handle_anthropic_tools,
                Mode.ANTHROPIC_REASONING_TOOLS: handle_anthropic_reasoning_tools,
                Mode.ANTHROPIC_JSON: handle_anthropic_json,
            }
        )

    if importlib.util.find_spec("cohere") is not None:
        mode_handlers.update(
            {
                Mode.COHERE_JSON_SCHEMA: handle_cohere_json_schema,
                Mode.COHERE_TOOLS: handle_cohere_tools,
            }
        )

    if importlib.util.find_spec("fireworks") is not None:
        mode_handlers.update(
            {
                Mode.FIREWORKS_JSON: handle_fireworks_json,
                Mode.FIREWORKS_TOOLS: handle_fireworks_tools,
            }
        )

    if importlib.util.find_spec("writerai") is not None:
        mode_handlers.update(
            {
                Mode.WRITER_TOOLS: handle_writer_tools,
                Mode.WRITER_JSON: handle_writer_json,
            }
        )

    if importlib.util.find_spec("openai") is not None:
        mode_handlers.update({Mode.PERPLEXITY_JSON: handle_perplexity_json})

    if importlib.util.find_spec("mistralai") is not None:
        pass  # already included above

    if importlib.util.find_spec("google.generativeai") is not None:
        mode_handlers.update(
            {
                Mode.GEMINI_JSON: handle_gemini_json,
                Mode.GEMINI_TOOLS: handle_gemini_tools,
            }
        )

    if importlib.util.find_spec("google.genai") is not None:
        mode_handlers.update(
            {
                Mode.GENAI_TOOLS: handle_genai_tools,
                Mode.GENAI_STRUCTURED_OUTPUTS: handle_genai_structured_outputs,
            }
        )

    if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
        mode_handlers.update(
            {
                Mode.VERTEXAI_TOOLS: handle_vertexai_tools,
                Mode.VERTEXAI_JSON: handle_vertexai_json,
                Mode.VERTEXAI_PARALLEL_TOOLS: handle_vertexai_parallel_tools,
            }
        )

    if importlib.util.find_spec("boto3") is not None:
        mode_handlers.update(
            {
                Mode.BEDROCK_JSON: handle_bedrock_json,
                Mode.BEDROCK_TOOLS: handle_bedrock_tools,
            }
        )

    if importlib.util.find_spec("cerebras") is not None:
        mode_handlers.update(
            {
                Mode.CEREBRAS_JSON: handle_cerebras_json,
                Mode.CEREBRAS_TOOLS: handle_cerebras_tools,
            }
        )

    if mode in mode_handlers:
        response_model, new_kwargs = mode_handlers[mode](response_model, new_kwargs)
    else:
        raise ValueError(f"Invalid patch mode: {mode}")

    if "messages" in new_kwargs:
        new_kwargs["messages"] = convert_messages(
            new_kwargs["messages"],
            mode,
            autodetect_images=autodetect_images,
        )

    if mode in {Mode.GENAI_TOOLS, Mode.GENAI_STRUCTURED_OUTPUTS}:
        new_kwargs["contents"] = extract_genai_multimodal_content(
            new_kwargs["contents"], autodetect_images
        )

    logger.debug(
        f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}",
        extra={
            "mode": mode.value,
            "response_model": (
                response_model.__name__
                if response_model is not None and hasattr(response_model, "__name__")
                else str(response_model)
            ),
            "new_kwargs": new_kwargs,
        },
    )
    return response_model, new_kwargs
