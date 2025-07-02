from __future__ import annotations

"""Claude Code provider integration for the *instructor* library.

This helper wraps the `claude-code-sdk` asynchronous generator so it can be
used as a drop-in provider inside *instructor*-powered projects.

Key features
------------
1. Converts the usual ChatGPT-style list-of-dict *messages* into a single prompt
   accepted by Claude Code.
2. Appends a system instruction that forces Claude to return **only** valid
   JSON matching the supplied Pydantic model schema (wrapped in a fenced
   ```json code-block for extra robustness).
3. Streams the response, extracts the JSON payload and converts it into the
   `response_model` instance (or raises helpful errors).
4. Offers both *async* (`acompletion`, `acreate`) and *sync* (`completion`,
   `create`) helpers so that it can be used alongside the rest of *instructor*.

Example
~~~~~~~
```python
from pydantic import BaseModel
from provider_claude import ClaudeCodeProvider

class Haiku(BaseModel):
    title: str
    poem: str
    author: str

provider = ClaudeCodeProvider()
result: Haiku = provider.create(
    prompt="Write a haiku about foo.py",
    response_model=Haiku,
)
print(result.json(indent=2))
```
"""

from typing import Any, List, Dict, Sequence, Type, Union, Iterable
import json
import anyio

from pydantic import BaseModel, ValidationError

try:
    # The SDK is optional; import lazily so that the rest of the codebase
    # remains usable even if the dependency is missing.
    from claude_code_sdk import query, ClaudeCodeOptions, Message  # type: ignore
except ImportError as e:  # pragma: no cover – dependency not present during tests
    raise ImportError(
        "`claude-code-sdk` is required for `ClaudeCodeProvider`.\n"
        "Install it with `pip install claude-code-sdk`"
    ) from e

from instructor.utils import extract_json_from_codeblock

__all__ = [
    "ClaudeCodeProvider",
]


class ClaudeCodeProvider:
    """Minimal provider that bridges Claude Code with *instructor*.

    Parameters
    ----------
    default_options:
        Base set of options passed to Claude Code for every request. Individual
        calls can override or extend these via the *options* kwarg.
    """

    def __init__(self, default_options: ClaudeCodeOptions | None = None):
        # If the caller doesn't supply a default we'll configure something that
        # is sensible for structured-output use-cases.
        self._default_options: ClaudeCodeOptions = default_options or ClaudeCodeOptions(
            max_turns=3,
            temperature=0.0,
        )

    # ---------------------------------------------------------------------
    # Public helpers -------------------------------------------------------
    # ---------------------------------------------------------------------
    async def acompletion(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        response_model: Type[BaseModel] | None = None,
        options: ClaudeCodeOptions | None = None,
        **extra_query_kwargs: Any,
    ) -> Union[BaseModel, str, Any]:
        """Asynchronously complete a list-of-messages conversation.

        Parameters
        ----------
        messages:
            A ChatGPT-style list of dicts with *role* and *content* keys.
        response_model:
            Pydantic model describing the expected JSON payload. If *None*, the
            raw text response is returned.
        options:
            Optional :class:`~claude_code_sdk.ClaudeCodeOptions` overriding the
            provider's defaults.
        **extra_query_kwargs:
            Forwarded as-is to :func:`claude_code_sdk.query`.
        """
        # 1. Flatten messages into a single prompt string
        prompt = self._messages_to_prompt(list(messages))

        # 2. If a response model is supplied, enforce JSON output
        if response_model is not None:
            prompt = self._append_json_instruction(prompt, response_model)

        # 3. Decide which options to use (caller overrides provider default)
        opts = options or self._default_options

        # 4. Collect streamed assistant chunks
        assistant_chunks: List[str] = []
        async for msg in query(prompt=prompt, options=opts, **extra_query_kwargs):
            # The SDK exposes different shapes depending on version. We try the
            # most common attribute names in order of preference.
            if getattr(msg, "kind", None) != "assistant":
                # Skip non-assistant events (e.g., system or user echoes)
                continue
            chunk_text = (
                getattr(msg, "text", None)
                or getattr(msg, "content", None)
                or str(msg)
            )
            assistant_chunks.append(chunk_text)

        full_response: str = "".join(assistant_chunks).strip()

        # 5. If no model, return raw text immediately
        if response_model is None:
            return full_response

        # 6. Extract & parse JSON
        json_str = extract_json_from_codeblock(full_response)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Claude Code did not return valid JSON.\n"
                f"Error: {exc}\nResponse:\n{full_response}"
            ) from exc

        # 7. Validate against schema
        try:
            return response_model.model_validate(data)  # type: ignore[attr-defined]
        except ValidationError as exc:
            raise ValueError(
                "Returned JSON does not conform to the response model.\n"
                f"Error: {exc}"
            ) from exc

    async def acreate(
        self,
        *,
        prompt: str,
        response_model: Type[BaseModel] | None = None,
        options: ClaudeCodeOptions | None = None,
        **extra_query_kwargs: Any,
    ) -> Union[BaseModel, str, Any]:
        """Helper that builds the *messages* list for single-prompt use-cases."""
        messages = [{"role": "user", "content": prompt}]
        return await self.acompletion(
            messages,
            response_model=response_model,
            options=options,
            **extra_query_kwargs,
        )

    # -----------------------
    # Synchronous wrappers
    # -----------------------
    def completion(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        response_model: Type[BaseModel] | None = None,
        options: ClaudeCodeOptions | None = None,
        **extra_query_kwargs: Any,
    ) -> Union[BaseModel, str, Any]:
        """Synchronous wrapper around :py:meth:`acompletion`."""
        return anyio.run(
            self.acompletion,
            list(messages),
            response_model=response_model,
            options=options,
            **extra_query_kwargs,
        )

    def create(
        self,
        *,
        prompt: str,
        response_model: Type[BaseModel] | None = None,
        options: ClaudeCodeOptions | None = None,
        **extra_query_kwargs: Any,
    ) -> Union[BaseModel, str, Any]:
        """Synchronous helper that mirrors :py:meth:`acreate`."""
        return anyio.run(
            self.acreate,
            prompt=prompt,
            response_model=response_model,
            options=options,
            **extra_query_kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert chat messages into a single prompt string.

        The strategy is simple: system messages are prefixed with `[System]` to
        preserve intent, all others are concatenated in the order they appear.
        """
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}\n")
            elif role == "user":
                parts.append(content)
            else:
                parts.append(f"[{role}]\n{content}\n")
        return "\n".join(parts).strip()

    @staticmethod
    def _append_json_instruction(prompt: str, model: Type[BaseModel]) -> str:
        """Attach instruction & JSON schema to *prompt* forcing valid output."""
        schema = json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False)
        instruction = (
            "\n\n# Instruction\n"
            "You MUST respond with **only** valid JSON that conforms to the following schema.\n"
            "The JSON *must* be wrapped in a fenced ```json code-block.\n\n"
            f"{schema}\n"
        )
        return prompt + instruction