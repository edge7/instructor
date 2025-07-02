#!/usr/bin/env python3
"""
Demo script for promptframe - Quick Start Example

This script demonstrates the basic usage of promptframe for enriching
DataFrames with LLM-derived insights.
"""

import pandas as pd
from pydantic import BaseModel, Field
from promptframe import PromptFrame


class Analysis(BaseModel):
    """Schema for text analysis results."""
    summary: str = Field(description="Brief summary of the text")
    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    tags: list[str] = Field(description="Relevant tags for the text")


class ProductAnalysis(BaseModel):
    """Schema for product analysis results."""
    category: str = Field(description="Product category")
    price_range: str = Field(description="Price range: low, medium, high")
    recommended: bool = Field(description="Whether to recommend this product")
    reasoning: str = Field(description="Brief reasoning for the recommendation")


def demo_basic_usage():
    """Demonstrate basic promptframe usage."""
    print("🚀 PromptFrame Demo - Basic Usage")
    print("=" * 50)
    
    # Sample data
    df = pd.DataFrame({
        "text": [
            "I absolutely love this new phone! The camera quality is amazing.",
            "This restaurant was terrible. The service was slow and food was cold.",
            "The weather today is quite nice, not too hot or cold.",
            "Just finished reading an excellent book about machine learning.",
        ]
    })
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Create PromptFrame and apply analysis
    pf = PromptFrame(df)
    
    pf.map_prompt(
        name="analysis",
        template="""
        Text: {{ text }}
        
        Analyze this text and provide:
        - A brief summary (1-2 sentences)
        - The overall sentiment (positive, negative, or neutral)
        - 2-3 relevant tags that describe the content
        """,
        schema=Analysis,
        progress=True
    )
    
    # Get enriched DataFrame
    enriched = pf.to_df()
    
    print("Enriched DataFrame:")
    print(enriched)
    print()
    
    # Show individual columns
    print("Analysis Results:")
    for i, row in enriched.iterrows():
        print(f"Row {i + 1}:")
        print(f"  Text: {row['text'][:50]}...")
        print(f"  Summary: {row['analysis.summary']}")
        print(f"  Sentiment: {row['analysis.sentiment']}")
        print(f"  Tags: {row['analysis.tags']}")
        print()


def demo_advanced_template():
    """Demonstrate advanced template usage with variables."""
    print("🛠️ PromptFrame Demo - Advanced Templates")
    print("=" * 50)
    
    # Product data
    df = pd.DataFrame({
        "name": ["iPhone 15", "Budget Laptop", "Gaming Chair", "Coffee Maker"],
        "description": [
            "Latest smartphone with advanced camera",
            "Basic laptop for everyday tasks",
            "Ergonomic chair for long gaming sessions",
            "Automatic drip coffee maker"
        ],
        "price": [999, 299, 199, 89]
    })
    
    print("Product DataFrame:")
    print(df)
    print()
    
    pf = PromptFrame(df)
    
    pf.map_prompt(
        name="product_analysis",
        template="""
        Product: {{ name }}
        Description: {{ description }}
        Price: ${{ price }}
        
        Context: This analysis is for {{ target_audience }} customers 
        with a budget preference of {{ budget_pref }}.
        
        Analyze this product considering the target audience and budget preferences.
        Categorize it, determine if the price is low/medium/high for this audience,
        and recommend whether they should buy it.
        """,
        schema=ProductAnalysis,
        template_kwargs={
            "target_audience": "young professionals",
            "budget_pref": "mid-range"
        },
        progress=True
    )
    
    enriched = pf.to_df()
    
    print("Product Analysis Results:")
    for i, row in enriched.iterrows():
        print(f"{row['name']} (${row['price']}):")
        print(f"  Category: {row['product_analysis.category']}")
        print(f"  Price Range: {row['product_analysis.price_range']}")
        print(f"  Recommended: {row['product_analysis.recommended']}")
        print(f"  Reasoning: {row['product_analysis.reasoning']}")
        print()


def demo_custom_template_function():
    """Demonstrate custom template functions."""
    print("⚡ PromptFrame Demo - Custom Template Functions")
    print("=" * 50)
    
    # Mixed content data
    df = pd.DataFrame({
        "content": [
            "Quarterly sales increased by 15% this month",
            "The new marketing campaign launched successfully",
            "Customer satisfaction scores improved significantly",
            "Server downtime lasted 2 hours yesterday"
        ],
        "content_type": ["sales_report", "marketing_update", "customer_feedback", "incident_report"],
        "priority": ["high", "medium", "medium", "urgent"]
    })
    
    print("Content DataFrame:")
    print(df)
    print()
    
    def custom_template(row):
        """Custom template function that adapts based on content type."""
        content_type = row['content_type']
        priority = row['priority']
        content = row['content']
        
        if content_type == "incident_report":
            return f"""
            INCIDENT REPORT - {priority.upper()} PRIORITY
            Content: {content}
            
            Analyze this incident report and provide:
            - A brief summary focused on impact and resolution
            - Sentiment should reflect urgency/concern level
            - Tags should include technical and business impact terms
            """
        elif content_type == "sales_report":
            return f"""
            SALES ANALYSIS - {priority.upper()} PRIORITY
            Content: {content}
            
            Analyze this sales information and provide:
            - A summary highlighting key metrics and trends
            - Sentiment reflecting business performance
            - Tags related to sales performance and business metrics
            """
        else:
            return f"""
            BUSINESS UPDATE - {priority.upper()} PRIORITY
            Content: {content}
            
            Analyze this business update and provide:
            - A concise summary of the key points
            - Overall sentiment of the development
            - Relevant business and operational tags
            """
    
    pf = PromptFrame(df)
    
    pf.map_prompt(
        name="custom_analysis",
        template=custom_template,
        schema=Analysis,
        progress=True
    )
    
    enriched = pf.to_df()
    
    print("Custom Analysis Results:")
    for i, row in enriched.iterrows():
        print(f"{row['content_type'].replace('_', ' ').title()} ({row['priority']}):")
        print(f"  Content: {row['content']}")
        print(f"  Summary: {row['custom_analysis.summary']}")
        print(f"  Sentiment: {row['custom_analysis.sentiment']}")
        print(f"  Tags: {row['custom_analysis.tags']}")
        print()


def main():
    """Run all demos."""
    print("🎯 PromptFrame Demo Suite")
    print("=" * 50)
    print("This demo requires OPENAI_API_KEY to be set in environment")
    print()
    
    try:
        demo_basic_usage()
        print("\n" + "="*50 + "\n")
        
        demo_advanced_template()
        print("\n" + "="*50 + "\n")
        
        demo_custom_template_function()
        
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Installed required dependencies: pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()