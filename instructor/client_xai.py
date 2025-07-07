from __future__ import annotations

from typing import Any, overload
from instructor.client import Instructor, AsyncInstructor
from instructor.hooks import Hooks
from instructor.mode import Mode
from instructor.utils import Provider

try:
    import xai_sdk
    from xai_sdk import chat as xchat
except ImportError:  # pragma: no cover - optional dependency
    xai_sdk = None  # type: ignore
    xchat = None  # type: ignore


# Helper to convert OpenAI style messages to xai_sdk messages

def _to_xai_message(message: dict[str, Any]) -> xchat.chat_pb2.Message:
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(xchat.text(item))
            elif isinstance(item, dict) and item.get("type") == "image":
                url = item.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                parts.append(xchat.image(url))
            else:
                raise ValueError(f"Unsupported content item: {item}")
    else:
        parts = [xchat.text(content)]

    if role == "user":
        return xchat.user(*parts)
    elif role == "assistant":
        return xchat.assistant(*parts)
    elif role == "system":
        return xchat.system(*parts)
    elif role == "tool":
        assert isinstance(content, str)
        return xchat.tool_result(content)
    else:
        raise ValueError(f"Unsupported role: {role}")


def _convert_messages(messages: list[dict[str, Any]]) -> list[xchat.chat_pb2.Message]:
    return [_to_xai_message(m) for m in messages]


@overload
def from_xai(client: xai_sdk.Client, mode: Mode = Mode.XAI_JSON, **kwargs: Any) -> Instructor: ...


@overload
def from_xai(client: xai_sdk.aio.Client, mode: Mode = Mode.XAI_JSON, **kwargs: Any) -> AsyncInstructor: ...


def from_xai(
    client: xai_sdk.Client | xai_sdk.aio.Client,
    mode: Mode = Mode.XAI_JSON,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    valid_modes = {Mode.XAI_JSON, Mode.XAI_TOOLS}
    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(mode=str(mode), provider="xai", valid_modes=[str(m) for m in valid_modes])

    if xai_sdk is None:
        raise ImportError("xai-sdk is required for from_xai")

    def create_fn(
        *,
        response_model: type[Any] | None,
        messages: list[dict[str, Any]],
        hooks: Hooks | None = None,  # noqa: ARG001
        **kw: Any,
    ) -> Any:
        pb_messages = _convert_messages(messages)
        chat = client.chat.create(messages=pb_messages, **kw)
        if response_model is None:
            resp = chat.sample()
            return resp
        if mode == Mode.XAI_JSON:
            resp, parsed = chat.parse(response_model)
            parsed._raw_response = resp
            return parsed
        else:  # Mode.XAI_TOOLS
            tool = xchat.tool(
                name=response_model.__name__,
                description=response_model.__doc__ or "",
                parameters=response_model.model_json_schema(),
            )
            chat.proto.tools[:] = [tool]
            resp = chat.sample()
            if not resp.tool_calls:
                raise ValueError("No tool call returned")
            args = resp.tool_calls[0].function.arguments
            parsed = response_model.model_validate_json(args)
            parsed._raw_response = resp
            return parsed

    if isinstance(client, xai_sdk.aio.Client):
        async def async_create_fn(**kw: Any) -> Any:
            return create_fn(**kw)

        return AsyncInstructor(
            client=client,
            create=async_create_fn,
            provider=Provider.XAI,
            mode=mode,
            **kwargs,
        )
    else:
        return Instructor(
            client=client,
            create=create_fn,
            provider=Provider.XAI,
            mode=mode,
            **kwargs,
        )
