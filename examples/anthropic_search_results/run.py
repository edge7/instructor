"""
Anthropic Search Results with Citations Example

This example demonstrates how to use Anthropic's new search results feature
with proper citations. The search results feature enables natural citations
for RAG applications by providing search results with source attribution.

Note: This feature requires the beta header 'search-results-2025-06-09'
"""

import instructor
from pydantic import BaseModel
from typing import List, Dict, Any
import anthropic


# Initialize the client with Instructor
client = instructor.from_anthropic(anthropic.Anthropic())


class Citation(BaseModel):
    """Citation model for tracking sources"""
    source: str
    title: str
    cited_text: str


class ResearchResponse(BaseModel):
    """Response model with citations"""
    answer: str
    citations: List[Citation]
    confidence: float = 0.8


def create_search_result_block(source: str, title: str, content: str) -> Dict[str, Any]:
    """Helper function to create search result blocks in the correct format"""
    return {
        "type": "search_result",
        "source": source,
        "title": title,
        "content": [
            {
                "type": "text",
                "text": content
            }
        ],
        "citations": {
            "enabled": True
        }
    }


def example_1_direct_search_results():
    """
    Example 1: Direct search results as top-level content
    This method is useful for pre-fetched content or cached search results
    """
    print("=" * 50)
    print("Example 1: Direct Search Results")
    print("=" * 50)
    
    # Create search results directly in the user message
    search_results = [
        create_search_result_block(
            source="https://docs.anthropic.com/en/docs/build-with-claude/search-results",
            title="Anthropic Search Results Documentation",
            content="Search result content blocks enable natural citations with proper source attribution, bringing web search-quality citations to your custom applications. This feature is particularly powerful for RAG (Retrieval-Augmented Generation) applications where you need Claude to cite sources accurately."
        ),
        create_search_result_block(
            source="https://research.anthropic.com/ai-safety",
            title="AI Safety Research at Anthropic",
            content="Anthropic focuses on AI safety research including Constitutional AI, which aims to train AI systems to be helpful, harmless, and honest. The company was founded by former OpenAI researchers who wanted to focus specifically on AI safety and alignment."
        ),
        create_search_result_block(
            source="https://blog.instructor.com/ai-citations",
            title="Best Practices for AI Citations",
            content="When implementing citation systems, it's important to ensure accuracy between claims and sources. LLM-based citation verification can help validate that citations match the content being referenced."
        )
    ]
    
    # Add the text query at the end
    user_content = search_results + [
        {
            "type": "text", 
            "text": "Based on these search results, explain what Anthropic's search results feature does and how it relates to AI safety. Provide proper citations."
        }
    ]
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            # Note: The beta header would be used here in production
            # betas=["search-results-2025-06-09"],
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            response_model=ResearchResponse,
        )
        
        print(f"Answer: {response.answer}")
        print(f"\nCitations found: {len(response.citations)}")
        for i, citation in enumerate(response.citations, 1):
            print(f"{i}. {citation.title}")
            print(f"   Source: {citation.source}")
            print(f"   Cited: {citation.cited_text[:100]}...")
            
    except Exception as e:
        print(f"Note: This example requires the beta header 'search-results-2025-06-09'")
        print(f"Error: {e}")
        
        # Fallback demonstration without beta features
        print("\nFallback: Simulating the response structure...")
        simulated_response = ResearchResponse(
            answer="Anthropic's search results feature enables natural citations with proper source attribution for RAG applications. This connects to AI safety by ensuring accurate information retrieval and source verification, which is crucial for building trustworthy AI systems.",
            citations=[
                Citation(
                    source="https://docs.anthropic.com/en/docs/build-with-claude/search-results",
                    title="Anthropic Search Results Documentation", 
                    cited_text="Search result content blocks enable natural citations with proper source attribution"
                ),
                Citation(
                    source="https://research.anthropic.com/ai-safety",
                    title="AI Safety Research at Anthropic",
                    cited_text="Constitutional AI, which aims to train AI systems to be helpful, harmless, and honest"
                )
            ]
        )
        
        print(f"Simulated Answer: {simulated_response.answer}")
        print(f"\nSimulated Citations: {len(simulated_response.citations)}")
        for i, citation in enumerate(simulated_response.citations, 1):
            print(f"{i}. {citation.title}")
            print(f"   Source: {citation.source}")


def example_2_tool_based_search():
    """
    Example 2: Search results from tool calls
    This would be used for dynamic RAG applications where tools fetch content
    """
    print("\n" + "=" * 50)
    print("Example 2: Tool-Based Search Results (Conceptual)")
    print("=" * 50)
    
    # This is a conceptual example of how tool-based search would work
    # In practice, you'd implement custom tools that return search results
    
    print("Conceptual tool implementation:")
    print("""
def search_knowledge_base(query: str) -> List[Dict]:
    '''Tool that returns search results in the correct format'''
    # Your search logic here
    return [
        {
            "type": "search_result",
            "source": "https://company-docs.com/api-guide",
            "title": "API Documentation",
            "content": [
                {
                    "type": "text",
                    "text": "API key authentication is required for all requests..."
                }
            ],
            "citations": {"enabled": True}
        }
    ]
    """)
    
    print("\nThis would enable dynamic RAG where:")
    print("1. User asks a question")
    print("2. Claude calls the search tool")  
    print("3. Tool returns properly formatted search results")
    print("4. Claude uses results with automatic citations")


def example_3_advanced_citation_patterns():
    """
    Example 3: Advanced citation patterns and validation
    """
    print("\n" + "=" * 50)
    print("Example 3: Advanced Citation Patterns")
    print("=" * 50)
    
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
    
    # Simulate an advanced response
    advanced_response = AdvancedResearchResponse(
        summary="Anthropic's search results feature represents a significant advancement in RAG applications, providing web search-quality citations for custom content sources.",
        key_findings=[
            "Enables natural citations for any content source",
            "Supports both tool-based and direct content methods", 
            "Maintains citation accuracy with source attribution",
            "Reduces need for document-based citation workarounds"
        ],
        citations=[
            DetailedCitation(
                source="https://docs.anthropic.com/en/docs/build-with-claude/search-results",
                title="Anthropic Documentation",
                cited_text="web search-quality citations for your custom applications",
                search_result_index=0,
                confidence_score=0.95
            ),
            DetailedCitation(
                source="https://blog.instructor.com/citations", 
                title="Citation Best Practices",
                cited_text="citation verification can enhance data accuracy",
                search_result_index=2,
                confidence_score=0.85
            )
        ]
    )
    
    print(f"Summary: {advanced_response.summary}")
    print(f"\nKey Findings:")
    for i, finding in enumerate(advanced_response.key_findings, 1):
        print(f"{i}. {finding}")
        
    print(f"\nSources analyzed: {advanced_response.sources_count}")
    print(f"Citations generated: {len(advanced_response.citations)}")
    
    print(f"\nCitation Details:")
    for citation in advanced_response.citations:
        print(f"- {citation.title} (confidence: {citation.confidence_score})")
        print(f"  {citation.cited_text}")


def main():
    """Run all examples"""
    print("Anthropic Search Results with Citations Examples")
    print("================================================")
    print("Note: These examples demonstrate the search results feature")
    print("which requires the beta header 'search-results-2025-06-09'\n")
    
    example_1_direct_search_results()
    example_2_tool_based_search()
    example_3_advanced_citation_patterns()
    
    print("\n" + "=" * 50)
    print("Key Benefits of Search Results Feature:")
    print("=" * 50)
    print("✓ Natural citations for RAG applications")
    print("✓ Flexible integration (tool returns or top-level content)")
    print("✓ Proper source attribution")
    print("✓ No document workarounds needed")
    print("✓ Consistent citation format")
    print("✓ Web search-quality citations for any content")


if __name__ == "__main__":
    main()