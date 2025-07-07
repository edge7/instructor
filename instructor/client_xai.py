from __future__ import annotations

from typing import overload, Any, TYPE_CHECKING, Sequence

import importlib.util

# Optional dependency: the xAI SDK is exposed through an OpenAI-compatible
# interface.  We import it lazily so that importing *instructor* does not
# hard-require `openai` at runtime unless the user actually calls
# `from_xai`.
if TYPE_CHECKING:  # pragma: no cover
    # `openai` may not be installed in all development environments. Guard
    # these imports so static analysis tools still get symbols without
    # requiring the runtime package.
    try:
        import openai  # type: ignore
        from openai.types.chat import ChatCompletion  # type: ignore
        from openai.types import CompletionUsage  # type: ignore
    except ModuleNotFoundError:
        from types import ModuleType

        openai = ModuleType("openai")  # type: ignore

        class _Stub:  # noqa: D401
            """Fallback stub for missing attributes when openai isn't present."""

        ChatCompletion = CompletionUsage = _Stub  # type: ignore

    # Optional pydantic types for docstring examples / annotations.
    try:
        from pydantic import BaseModel  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        class BaseModel:  # type: ignore
            """Minimal stub when *pydantic* is not available."""
            pass

# Runtime import for the actual OpenAI client (if installed). We do this
# outside TYPE_CHECKING so production code can execute.
try:
    if importlib.util.find_spec("openai") is not None:
        import openai  # type: ignore

        # Error classes (used at runtime for error mapping). Guarded with
        # try/except in case a slimmed-down OpenAI client omits some classes.
        try:
            from openai.error import (  # type: ignore
                AuthenticationError as _OpenAIAuthError,  # type: ignore
                BadRequestError as _OpenAIBadRequest,  # type: ignore
                RateLimitError as _OpenAIRateLimit,  # type: ignore
                APIConnectionError as _OpenAIConnError,  # type: ignore
                APITimeoutError as _OpenAITimeoutError,  # type: ignore
                InternalServerError as _OpenAIServerError,  # type: ignore
                ServiceUnavailableError as _OpenAIUnavailableError,  # type: ignore
                OpenAIError as _OpenAIError,  # type: ignore
            )
        except Exception:  # pragma: no cover
            class _StubError(Exception):
                """Fallback stub for missing OpenAI error classes."""

            _OpenAIAuthError = _OpenAIBadRequest = _OpenAIRateLimit = _OpenAIConnError = _OpenAITimeoutError = _OpenAIServerError = _OpenAIUnavailableError = _OpenAIError = _StubError  # type: ignore
    else:  # openai not installed
        openai = None  # type: ignore
        # Create stub error classes so later references are defined.
        _OpenAIAuthError = _OpenAIBadRequest = _OpenAIRateLimit = _OpenAIConnError = _OpenAITimeoutError = _OpenAIServerError = _OpenAIUnavailableError = _OpenAIError = Exception  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Fully missing openai package.
    openai = None  # type: ignore
    _OpenAIAuthError = _OpenAIBadRequest = _OpenAIRateLimit = _OpenAIConnError = _OpenAITimeoutError = _OpenAIServerError = _OpenAIUnavailableError = _OpenAIError = Exception  # type: ignore

import instructor
from instructor.exceptions import ClientError, ModeError, ProviderError

__all__ = ["from_xai"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Officially documented xAI model prefixes. We keep this list minimal on
# purpose – it is only meant to catch obvious mis-configurations (e.g. trying
# to send an OpenAI model id to an xAI endpoint).
# Update this tuple whenever xAI publishes new model families.
_XAI_MODEL_PREFIXES: Sequence[str] = (
    "grok",  # grok-beta, grok-3-beta, etc.
    "x-ai/",  # vendor style e.g. "x-ai/grok-3-beta" used by some gateways
)

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
    """Wrap an *xAI* client with *Instructor*.

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

        *Streaming* is **not** yet implemented in this helper – set
        ``stream=False`` when calling ``create``.
    **kwargs
        Forwarded to :class:`~instructor.Instructor` or
        :class:`~instructor.AsyncInstructor`.

    Returns
    -------
    instructor.Instructor | instructor.AsyncInstructor
        A thin wrapper around your *xAI* client that automatically parses the
        model response into the supplied ``response_model`` when you call
        ``.chat.completions.create``.

    Supported Models
    ----------------
    * ``grok-beta`` – legacy model.
    * ``grok-3-beta`` / ``grok-3-mini-beta`` – current flagship (Spring 2025).

    Requirements & Configuration
    ----------------------------
    1. The client **must** be instantiated with::

           base_url="https://api.x.ai/v1"

    2. Provide a valid ``api_key`` – the same key used on the xAI console and
       exposed as the ``XAI_API_KEY`` environment variable.
    3. Only *non-streaming* requests are supported at the moment. Pass
       ``stream=False`` (default) when calling ``create``.

    xAI Rate Limits & Quotas
    -----------------------
    * **Free-tier** accounts – 20 requests / min, 40k tokens-in, 40k tokens-out per
      day.
    * **Starter** plan – 60 requests / min, 250k tokens-in, 250k tokens-out per
      day.
    * **Pro / Enterprise** – contact xAI sales for custom quotas.

    Hitting a quota limit results in HTTP 429 with an ``rate_limit_exceeded``
    error code – *Instructor* translates that into
    :class:`~instructor.exceptions.ProviderError` with a descriptive message.

    Known Limitations
    -----------------
    * Function/tool calling and JSON mode are supported. Parallel tool calls
      and other experimental modes are **not** yet validated.
    * Vision inputs, searches and other beta capabilities are outside the
      scope of this helper – use raw xAI API until Instructor gains support.

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

    # ---------------------------------------------------------------------
    # 1.  Validate that the default / provided model looks like an xAI model
    # ---------------------------------------------------------------------

    model_in_kwargs: str | None = kwargs.get("default_model") or kwargs.get("model")
    prefixes: tuple[str, ...] = tuple(_XAI_MODEL_PREFIXES)
    if model_in_kwargs is not None and not model_in_kwargs.startswith(prefixes):
        raise ClientError(
            "The provided `model` does not look like an xAI Grok model. "
            f"Got '{model_in_kwargs}'. Valid models start with one of: {', '.join(_XAI_MODEL_PREFIXES)}"
        )

    # ---------------------------------------------------------------------
    # 2.  Wrap the low-level create call to convert xAI-specific API errors
    #     into Instructor's ProviderError for consistent error handling.
    # ---------------------------------------------------------------------

    raw_create = client.chat.completions.create  # type: ignore[attr-defined]

    def _safe_create(*args: Any, **kw: Any) -> Any:  # noqa: D401 – simple wrapper
        """Delegate to the original create but normalize xAI API errors."""

        # If the caller explicitly sets an unsupported model id we can catch
        # it early. Otherwise we let the server decide.
        if "model" in kw and not kw["model"].startswith(prefixes):
            raise ClientError(
                f"Unsupported xAI model '{kw['model']}'. Models must start with one "
                f"of: {', '.join(_XAI_MODEL_PREFIXES)}"
            )

        try:
            return raw_create(*args, **kw)
        except (_OpenAIBadRequest) as exc:  # type: ignore[name-defined]
            # Token limit / model not found / invalid tool invocation etc.
            exc_msg = str(exc)
            if "maximum context length" in exc_msg or "token" in exc_msg:
                raise ProviderError("xAI", "Token limit exceeded for model context window.") from exc
            if "model" in exc_msg and "not found" in exc_msg:
                raise ProviderError("xAI", "Unsupported or unknown xAI model id.") from exc
            if "Invalid function call" in exc_msg or "Invalid tool" in exc_msg:
                raise ProviderError("xAI", "Invalid function/tool call format or arguments.") from exc
            raise ProviderError("xAI", f"Bad request: {exc}") from exc
        except (_OpenAIAuthError) as exc:  # type: ignore[name-defined]
            raise ProviderError(
                "xAI",
                "Authentication failed – check `XAI_API_KEY` and organisation settings.",
            ) from exc
        except (_OpenAIRateLimit) as exc:  # type: ignore[name-defined]
            raise ProviderError("xAI", "Rate limit exceeded – slow down your requests.") from exc
        except (_OpenAITimeoutError) as exc:  # type: ignore[name-defined]
            raise ProviderError("xAI", "Request timed-out – retry later or increase timeout.") from exc
        except (_OpenAIConnError) as exc:  # type: ignore[name-defined]
            raise ProviderError("xAI", "Network error talking to xAI – check connectivity.") from exc
        except (_OpenAIServerError | _OpenAIUnavailableError) as exc:  # type: ignore[name-defined]
            raise ProviderError("xAI", "xAI service is currently unavailable – please retry.") from exc
        except (_OpenAIError) as exc:  # type: ignore[name-defined]
            raise ProviderError("xAI", f"API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 – catch-all fallback
            raise ProviderError("xAI", f"Unexpected error: {exc}") from exc

    patched_create = instructor.patch(create=_safe_create, mode=mode)

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