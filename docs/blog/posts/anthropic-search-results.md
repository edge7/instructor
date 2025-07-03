---
date: 2025-07-03
authors:
  - jxnl
categories:
  - tutorials
  - anthropic
  - citations
---

# Natural Citations with Search Result Content Blocks in Anthropic

Anthropic just released **search result content blocks** (beta header `search-results-2025-06-09`).  
This feature lets you hand-feed Claude search results and have it **cite** them just like the built-in web search tool.

In this short post we will:

1. Explain when to use search result blocks instead of the web-search tool.
2. Walk through a minimal Python example that you can run with Instructor.
3. Show what the cited output looks like.

<!-- more -->

## Why Search Result Blocks?

The *web_search* tool is great when Claude needs to go fetch fresh data on its own.  
But sometimes **you already have the relevant documents** – for example results coming from:

* your private vector store,
* an internal SQL database, or
* a cached web crawler.

With search result blocks you can embed those snippets directly in the message and Claude will:

* read the text,
* answer the user's question, and
* add inline citations automatically.

No more string-parsing hacks to line up sources with answers.

## Hands-On Example

First install the latest SDK (needs `anthropic>=0.53.0`, adjust once the beta types land in stable):

```bash
uv add anthropic instructor
```

The example script is in `examples/anthropic-search-results/run.py`.  Here is the essence:

```python
from anthropic import Anthropic
from anthropic.types.beta import (
    BetaMessageParam,
    BetaTextBlockParam,
    BetaSearchResultBlockParam,
)

client = Anthropic()

user_message = BetaMessageParam(
    role="user",
    content=[
        BetaSearchResultBlockParam(
            type="search_result",
            source="https://docs.python.org/3/library/datetime.html",
            title="datetime — Basic date and time types",
            content=[
                BetaTextBlockParam(
                    type="text",
                    text="`datetime.timedelta` is a duration expressing the difference between two date, time, or datetime instances.",
                )
            ],
            citations={"enabled": True},
        ),
        BetaTextBlockParam(type="text", text="What does `datetime.timedelta` represent in Python?"),
    ],
)

response = client.beta.messages.create(
    model="claude-opus-4-20250514",
    betas=["search-results-2025-06-09"],
    max_tokens=256,
    messages=[user_message],
)

print(response.model_dump_json(indent=2))
```

Run it and you should see something like:

```
Claude's answer (truncated):
`datetime.timedelta` represents a **time duration** … [^1]

[^1]: https://docs.python.org/3/library/datetime.html
```

Notice how Claude automatically inserted the citation and linked it back to our provided `source`.

## Tips and Gotchas

* **All-or-nothing citations** – every search result block in the same request must have `citations.enabled` set to the same value.
* Split long passages into multiple text blocks for finer-grained citations.
* You can combine *web_search* **and** your own search result blocks in the same conversation.

## Wrap-Up

Search result content blocks close the gap between tool-initiated search and pre-fetched RAG pipelines.  
They give you web-search quality citations while keeping full control over what text Claude sees.

Give it a try and let us know what you build!  
This post's code lives in `examples/anthropic-search-results`.