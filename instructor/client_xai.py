from __future__ import annotations

from typing import overload, Any

import openai
import instructor


@overload
def from_xai(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_xai(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_xai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor or AsyncInstructor instance for the xAI SDK.

    Parameters
    ----------
    client:
        An instance of ``openai.OpenAI`` or ``openai.AsyncOpenAI`` that is
        configured with ``base_url='https://api.x.ai/v1'`` and a valid
        ``api_key``.
    mode:
        The :class:`instructor.Mode` to use when patching the ``create``
        method. Only :pyattr:`instructor.Mode.JSON` and
        :pyattr:`instructor.Mode.TOOLS` are currently supported for the xAI
        provider.
    **kwargs:
        Additional keyword arguments are forwarded to the
        :class:`instructor.Instructor` / :class:`instructor.AsyncInstructor`
        constructor.

    Returns
    -------
    instructor.Instructor | instructor.AsyncInstructor
        A patched client that routes ``chat.completions.create`` through
        *Instructor* for automatic response model parsing.
    """

    valid_modes = {
        instructor.Mode.JSON,
        instructor.Mode.TOOLS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="xAI", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        from instructor.exceptions import ClientError

        raise ClientError(
            "Client must be an instance of openai.OpenAI or openai.AsyncOpenAI. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, openai.OpenAI):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            provider=instructor.Provider.XAI,
            mode=mode,
            **kwargs,
        )