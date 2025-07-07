from __future__ import annotations

from typing import overload, Any, TYPE_CHECKING

import importlib.util

# Optional dependency: the xAI SDK is exposed through an OpenAI-compatible
# interface.  We import it lazily so that importing *instructor* does not
# hard-require `openai` at runtime unless the user actually calls
# `from_xai`.
if TYPE_CHECKING or importlib.util.find_spec("openai") is not None:  # pragma: no cover
    import openai  # type: ignore
else:  # pragma: no cover
    openai = None  # type: ignore

import instructor
from instructor.exceptions import ClientError, ModeError

__all__ = ["from_xai"]


@overload
def from_xai(
    client: "openai.OpenAI",  # type: ignore[name-defined]
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_xai(
    client: "openai.AsyncOpenAI",  # type: ignore[name-defined]
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_xai(
    client: Any,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Wrap an xAI client with *Instructor*.

    Parameters
    ----------
    client
        An ``openai.OpenAI`` **or** ``openai.AsyncOpenAI`` instance *already*
        configured for xAI, e.g.::

            import openai, instructor

            xai_client = openai.OpenAI(
                api_key="<XAI_API_KEY>",
                base_url="https://api.x.ai/v1",  # NOTE: required
            )

            wrapped = instructor.from_xai(xai_client)

        The helper will **not** mutate global OpenAI settings; it simply
        patches the given instance.
    mode
        :class:`instructor.Mode` specifying how responses should be parsed.
        Currently xAI supports two modes:

        * ``Mode.JSON`` – forces *JSON mode* on Grok models.
        * ``Mode.TOOLS`` – enables function / tool calling for structured
          output.

        Streaming is *not* yet implemented in this helper.
    **kwargs
        Forwarded to :class:`~instructor.Instructor` or
        :class:`~instructor.AsyncInstructor`.

    Returns
    -------
    instructor.Instructor | instructor.AsyncInstructor
        A thin wrapper around your *xAI* client that automatically parses the
        model response into the supplied ``response_model`` when you call
        ``.chat.completions.create``.

    Examples
    --------
    ```python
    from pydantic import BaseModel
    import openai, instructor

    class Joke(BaseModel):
        setup: str
        punchline: str

    client = openai.OpenAI(api_key="sk-...", base_url="https://api.x.ai/v1")
    grok = instructor.from_xai(client, mode=instructor.Mode.TOOLS)

    joke: Joke = grok.chat.completions.create(
        model="grok-3-beta",
        messages=[{"role": "user", "content": "Tell me a joke about cats."}],
        response_model=Joke,
    )
    ```
    """

    if openai is None:  # pragma: no cover
        raise ImportError(
            "`openai` package is required to use from_xai(). Install it first."
        )

    valid_modes = {instructor.Mode.JSON, instructor.Mode.TOOLS}
    if mode not in valid_modes:
        raise ModeError(
            mode=str(mode), provider="xAI", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):  # type: ignore[attr-defined]
        raise ClientError(
            "Client must be an instance of openai.OpenAI or openai.AsyncOpenAI. "
            f"Got: {type(client).__name__}"
        )

    # Basic sanity-check: ensure the client's base_url points to the xAI API.
    base_url: str | None = getattr(client, "base_url", None)  # type: ignore[attr-defined]
    if base_url is None or "x.ai" not in base_url:
        raise ClientError(
            "The provided OpenAI client is not configured for xAI. "
            "Make sure to set `base_url='https://api.x.ai/v1'`."
        )

    patched_create = instructor.patch(create=client.chat.completions.create, mode=mode)

    if isinstance(client, openai.OpenAI):  # type: ignore[attr-defined]
        return instructor.Instructor(
            client=client,
            create=patched_create,
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )

    # Async path
    return instructor.AsyncInstructor(
        client=client,
        create=patched_create,
        provider=instructor.Provider.XAI,
        mode=mode,
        **kwargs,
    )