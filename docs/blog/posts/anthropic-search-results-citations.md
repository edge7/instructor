---
date: 2025-06-09
authors:
  - jxnl
categories:
  - tutorials
  - anthropic
  - citations
  - rag
---

# Anthropic's Search Results Feature: Natural Citations for RAG Applications

Anthropic has introduced a powerful new search results feature that brings web search-quality citations to your custom RAG applications. Combined with Instructor's structured data capabilities, this enables natural citations with proper source attribution for any content source.

In this post, we'll explore how to use the new search results content blocks to build citation-aware applications that maintain accuracy and traceability in their responses.

<!-- more -->

## What's New: Search Results Content Blocks

The search results feature introduces a new content block type that enables natural citations with proper source attribution. This is particularly powerful for RAG (Retrieval-Augmented Generation) applications where you need Claude to cite sources accurately.

### Key Benefits

- **Natural Citations**: Achieve the same citation quality as web search for any content
- **Flexible Integration**: Use in tool returns for dynamic RAG or as top-level content for pre-fetched data  
- **Proper Source Attribution**: Each result includes source and title information for clear attribution
- **No Document Workarounds**: Eliminates the need for document-based citation workarounds
- **Consistent Citation Format**: Matches the citation quality and format of Claude's web search functionality

## How It Works

Search results can be provided in two ways:

1. **From tool calls** - Your custom tools return search results, enabling dynamic RAG applications
2. **As top-level content** - You provide search results directly in user messages for pre-fetched or cached content

In both cases, Claude can automatically cite information from the search results with proper source attribution.

## Getting Started

First, ensure you have the required dependencies:

```bash
uv add instructor anthropic
```

The search results feature requires the beta header `search-results-2025-06-09`:

```python
import instructor
from pydantic import BaseModel
from typing import List
import anthropic

client = instructor.from_anthropic(anthropic.Anthropic())

class Citation(BaseModel):
    source: str
    title: str
    cited_text: str

class ResearchResponse(BaseModel):
    answer: str
    citations: List[Citation]
```

## Method 1: Direct Search Results

The simplest approach is providing search results directly in user messages. This is ideal for pre-fetched content, cached search results, or content from external search services.

```python
def create_search_result_block(source: str, title: str, content: str):
    """Helper function to create search result blocks"""
    return {
        "type": "search_result",
        "source": source,
        "title": title,
        "content": [{"type": "text", "text": content}],
        "citations": {"enabled": True}
    }

# Create search results
search_results = [
    create_search_result_block(
        source="https://docs.anthropic.com/en/docs/build-with-claude/search-results",
        title="Anthropic Search Results Documentation",
        content="Search result content blocks enable natural citations with proper source attribution, bringing web search-quality citations to your custom applications."
    ),
    create_search_result_block(
        source="https://research.anthropic.com/ai-safety",
        title="AI Safety Research at Anthropic", 
        content="Anthropic focuses on AI safety research including Constitutional AI, which aims to train AI systems to be helpful, harmless, and honest."
    )
]

# Combine with text query
user_content = search_results + [
    {
        "type": "text",
        "text": "Based on these search results, explain Anthropic's search results feature and how it relates to AI safety. Provide proper citations."
    }
]

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    betas=["search-results-2025-06-09"],  # Required beta header
    messages=[{"role": "user", "content": user_content}],
    response_model=ResearchResponse,
)

print(f"Answer: {response.answer}")
for citation in response.citations:
    print(f"Citation: {citation.title} - {citation.source}")
```

## Method 2: Tool-Based Search Results

The most powerful use case is returning search results from your custom tools. This enables dynamic RAG applications where tools fetch and return relevant content with automatic citations.

```python
def search_knowledge_base(query: str):
    """Custom tool that returns search results in the correct format"""
    # Your search logic here - could query vector databases,
    # search APIs, document stores, etc.
    
    return [
        {
            "type": "search_result",
            "source": "https://company-docs.com/api-reference",
            "title": "API Reference - Authentication",
            "content": [
                {
                    "type": "text",
                    "text": "All API requests must include an API key in the Authorization header. Keys can be generated from the dashboard. Rate limits: 1000 requests per hour for standard tier, 10000 for premium."
                }
            ],
            "citations": {"enabled": True}
        },
        {
            "type": "search_result",
            "source": "https://company-docs.com/quickstart",
            "title": "Getting Started Guide",
            "content": [
                {
                    "type": "text", 
                    "text": "To get started: 1) Sign up for an account, 2) Generate an API key from the dashboard, 3) Install our SDK using pip install company-sdk, 4) Initialize the client with your API key."
                }
            ],
            "citations": {"enabled": True}
        }
    ]

# Define the tool for Claude to use
knowledge_base_tool = {
    "name": "search_knowledge_base",
    "description": "Search the company knowledge base for information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
}

# Use with tool calling
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    betas=["search-results-2025-06-09"],
    tools=[knowledge_base_tool],
    messages=[
        {
            "role": "user",
            "content": "How do I authenticate API requests and what are the rate limits?"
        }
    ]
)

# When Claude calls the tool, provide the search results
if response.content[0].type == "tool_use":
    tool_result = search_knowledge_base(response.content[0].input["query"])
    
    # Send the tool result back
    final_response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        betas=["search-results-2025-06-09"],
        messages=[
            {"role": "user", "content": "How do I authenticate API requests and what are the rate limits?"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": response.content[0].id,
                        "content": tool_result  # Search results go here
                    }
                ]
            }
        ]
    )
```

## Citation Response Format

When Claude uses information from search results, it automatically includes detailed citation information:

```python
{
    "type": "search_result_location",
    "source": "https://docs.company.com/api-reference",
    "title": "API Reference - Authentication", 
    "cited_text": "All API requests must include an API key in the Authorization header",
    "search_result_index": 0,
    "start_block_index": 0,
    "end_block_index": 0
}
```

Each citation includes:
- **source**: The source from the original search result
- **title**: The title from the original search result  
- **cited_text**: The exact text being cited
- **search_result_index**: Index of the search result (0-based)
- **start_block_index/end_block_index**: Position in the content array

## Advanced Usage Patterns

### Enhanced Citation Models

You can create more sophisticated citation models with additional validation:

```python
class DetailedCitation(BaseModel):
    """Enhanced citation model with validation"""
    source: str
    title: str
    cited_text: str
    search_result_index: int
    confidence_score: float = 0.9
    
class AdvancedResearchResponse(BaseModel):
    """Advanced response with detailed citations"""
    summary: str
    key_findings: List[str]
    citations: List[DetailedCitation]
    sources_count: int
    
    def __init__(self, **data):
        super().__init__(**data)
        self.sources_count = len(set(c.source for c in self.citations))
```

### Multiple Content Blocks

Search results can contain multiple text blocks for better organization:

```python
{
    "type": "search_result",
    "source": "https://docs.company.com/api-guide",
    "title": "API Documentation",
    "content": [
        {"type": "text", "text": "Authentication: All API requests require an API key."},
        {"type": "text", "text": "Rate Limits: The API allows 1000 requests per hour per key."},
        {"type": "text", "text": "Error Handling: The API returns standard HTTP status codes."}
    ]
}
```

Claude can cite specific blocks using the `start_block_index` and `end_block_index` fields.

### Combining with Other Content Types

Both methods support mixing search results with other content:

```python
# In user messages
user_content = [
    search_result_block,  # Search result
    {"type": "image", "source": {"type": "url", "url": "https://example.com/chart.png"}},
    {"type": "text", "text": "How does the chart relate to the research findings?"}
]
```

## Schema Reference

### Required Fields
- `type`: Must be "search_result"
- `source`: The source URL or identifier for the content
- `title`: A descriptive title for the search result  
- `content`: An array of text blocks containing the actual content

### Optional Fields
- `citations`: Citation configuration with `enabled` boolean field
- `cache_control`: Cache control settings (e.g., `{"type": "ephemeral"}`)

Each item in the content array must be a text block with:
- `type`: Must be "text"
- `text`: The actual text content (non-empty string)

## Best Practices

### For Tool-Based Search (Method 1)
- **Dynamic content**: Use for real-time searches and dynamic RAG applications
- **Error handling**: Return appropriate messages when searches fail
- **Result limits**: Return only the most relevant results to avoid context overflow

### For Direct Search (Method 2)  
- **Pre-fetched content**: Use when you already have search results
- **Batch processing**: Ideal for processing multiple search results at once
- **Testing**: Great for testing citation behavior with known content

### General Best Practices
1. **Structure results effectively**
   - Use clear, permanent source URLs
   - Provide descriptive titles
   - Break long content into logical text blocks

2. **Maintain consistency**
   - Use consistent source formats across your application
   - Ensure titles accurately reflect content
   - Keep formatting consistent

3. **Handle errors gracefully**
   ```python
   def search_with_fallback(query):
       try:
           results = perform_search(query)
           if not results:
               return {"type": "text", "text": "No results found."}
           return format_as_search_results(results)
       except Exception as e:
           return {"type": "text", "text": f"Search error: {str(e)}"}
   ```

## Citation Control

By default, citations are disabled for search results. You must explicitly enable them:

```python
{
    "type": "search_result",
    "source": "https://docs.company.com/guide",
    "title": "User Guide", 
    "content": [{"type": "text", "text": "Important documentation..."}],
    "citations": {
        "enabled": True  # Must be explicitly enabled
    }
}
```

**Important**: Citations are all-or-nothing within a request. Either all search results must have citations enabled, or all must have them disabled. Mixing different citation settings will result in an error.

## Integration with Existing Citation Patterns

The search results feature works well with existing citation validation patterns from Instructor:

```python
from pydantic import field_validator, ValidationInfo

class ValidatedCitation(BaseModel):
    source: str
    title: str
    cited_text: str
    
    @field_validator("cited_text")
    @classmethod
    def validate_citation_exists(cls, v: str, info: ValidationInfo):
        # Custom validation logic here
        context = info.context.get("search_results", [])
        # Verify citation exists in search results
        return v
```

## Limitations

- Search result content blocks are only available with the beta header
- Only text content is supported within search results (no images or other media)
- The content array must contain at least one text block
- Citations are all-or-nothing per request

## Conclusion

Anthropic's search results feature represents a significant advancement in RAG applications, providing web search-quality citations for any content source. When combined with Instructor's structured data capabilities, it enables building citation-aware applications that maintain accuracy and traceability.

Whether you're building knowledge bases, research tools, or documentation systems, the search results feature provides the citation infrastructure needed for trustworthy AI applications.

## Related Documentation

- [Citation Validation with Pydantic](citations.md) - Validate citations with LLMs
- [RAG Techniques](rag-and-beyond.md) - Advanced RAG patterns
- [Anthropic Web Search](anthropic-web-search-structured.md) - Web search with Instructor

Check out the example code in `examples/anthropic_search_results/run.py` to see this implementation in action.

If you like this content, check out our [GitHub](https://github.com/jxnl/instructor) and give us a star!