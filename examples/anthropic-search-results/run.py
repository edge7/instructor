"""
Example: Using Anthropic Search Result Content Blocks

This script shows how to provide your own search results to Claude so that it can
cite them naturally in its response.  This mirrors the example from the
Anthropic blog post published on **2025-06-09** and requires the beta header
`search-results-2025-06-09`.

NOTE  ▸  You need an `ANTHROPIC_API_KEY` exported in your environment for this
script to run.  In CI we only keep the code as reference – it will **not**
execute without valid credentials.
"""

from anthropic import Anthropic  # type: ignore
from anthropic.types.beta import (  # type: ignore
    BetaMessageParam,
    BetaTextBlockParam,
    BetaSearchResultBlockParam,
)

# Create a standard synchronous client
client = Anthropic()

# ----- Build the message with *inline* search-result blocks ---------------
question = "What does `datetime.timedelta` represent in Python? Give a short answer."

user_message = BetaMessageParam(
    role="user",
    content=[
        # First search result
        BetaSearchResultBlockParam(
            type="search_result",
            source="https://docs.python.org/3/library/datetime.html",
            title="datetime — Basic date and time types",
            content=[
                BetaTextBlockParam(
                    type="text",
                    text=(
                        "The datetime module supplies classes for manipulating "
                        "dates and times.  `datetime.timedelta` is a duration "
                        "expressing the difference between two *date*, *time*, "
                        "or *datetime* instances."
                    ),
                )
            ],
            citations={"enabled": True},
        ),
        # You can include any number of results – here we stick with one.
        BetaTextBlockParam(type="text", text=question),
    ],
)

# ----- Call the beta API ---------------------------------------------------
response = client.beta.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=256,
    betas=["search-results-2025-06-09"],
    messages=[user_message],
)

print("Claude's answer (with citations):")
print(response.model_dump_json(indent=2))