"""Minimal demo for using ClaudeCodeProvider with instructor.

NOTE: Running this script requires:
    $ pip install instructor[claude-code]
    $ npm install -g @anthropic-ai/claude-code
and a working Node.js installation.

Replace the `prompt` below with anything you like. The example expects Claude
Code to reply with a JSON object describing a haiku.
"""

import anyio
from pydantic import BaseModel
from typing import cast

from provider_claude import ClaudeCodeProvider


class Haiku(BaseModel):
    title: str
    poem: str
    author: str


async def main() -> None:
    provider = ClaudeCodeProvider()

    haiku: Haiku = cast(
        Haiku,
        await provider.acreate(
            prompt="Write a haiku about foo.py",
            response_model=Haiku,
        ),
    )
    print(haiku.model_dump_json(indent=2))


if __name__ == "__main__":
    anyio.run(main)