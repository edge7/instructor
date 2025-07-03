# Anthropic Search Results with Citations

This example demonstrates how to use Anthropic's new search results feature with proper citations. The search results feature enables natural citations for RAG (Retrieval-Augmented Generation) applications by providing search results with source attribution.

## Key Features

- **Natural Citations**: Achieve web search-quality citations for any content
- **Flexible Integration**: Use in tool returns for dynamic RAG or as top-level content for pre-fetched data
- **Proper Source Attribution**: Each result includes source and title information for clear attribution
- **No Document Workarounds**: Eliminates the need for document-based citation workarounds
- **Consistent Citation Format**: Matches the citation quality and format of Claude's web search functionality

## Requirements

This feature is currently in beta and requires:
- The `search-results-2025-06-09` beta header
- Anthropic API client with instructor

## Installation

```bash
pip install instructor anthropic
```

## Usage

The example demonstrates three approaches:

### 1. Direct Search Results (Method 1)
```python
search_results = [
    {
        "type": "search_result",
        "source": "https://example.com/source",
        "title": "Document Title",
        "content": [{"type": "text", "text": "Content here..."}],
        "citations": {"enabled": True}
    }
]
```

### 2. Tool-Based Search Results (Method 2)
For dynamic RAG applications where tools fetch and return relevant content with automatic citations.

### 3. Advanced Citation Patterns
Enhanced citation models with validation and confidence scoring.

## Running the Example

```bash
cd examples/anthropic_search_results
python run.py
```

## Example Output

The example will show:
- How to structure search results for citations
- Automatic citation generation by Claude
- Different citation patterns and validation approaches
- Best practices for RAG applications

## Schema Reference

### Search Result Block Structure
```python
{
    "type": "search_result",          # Required: Must be "search_result"
    "source": "string",               # Required: Source URL or identifier
    "title": "string",                # Required: Title of the result
    "content": [                      # Required: Array of text blocks
        {
            "type": "text",           # Required: Must be "text"
            "text": "string"          # Required: Actual content
        }
    ],
    "citations": {                    # Optional: Citation configuration
        "enabled": true               # Enable/disable citations
    }
}
```

### Citation Response Format
Claude automatically includes citations when using information from search results:
```python
{
    "type": "search_result_location",
    "source": "https://example.com",
    "title": "Document Title",
    "cited_text": "The exact text being cited",
    "search_result_index": 0,
    "start_block_index": 0,
    "end_block_index": 0
}
```

## Best Practices

1. **Use clear, permanent source URLs**
2. **Provide descriptive titles**
3. **Break long content into logical text blocks**
4. **Enable citations consistently** (all-or-nothing within a request)
5. **Handle errors gracefully** for production applications

## Related Documentation

- [Anthropic Search Results Documentation](https://docs.anthropic.com/en/docs/build-with-claude/search-results)
- [Instructor Citations Guide](../../docs/blog/posts/citations.md)
- [RAG Best Practices](../../docs/blog/posts/rag-and-beyond.md)

## Limitations

- Search result content blocks are only available with the beta header
- Only text content is supported within search results (no images or other media)
- The content array must contain at least one text block
- Citations are all-or-nothing per request (cannot mix enabled/disabled within same request)