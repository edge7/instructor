#!/usr/bin/env python3
"""
Simple example demonstrating promptframe library structure and basic functionality.

This example shows how the library works without making actual API calls.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pydantic import BaseModel, Field


def main():
    """Demonstrate the basic structure and imports of promptframe."""
    print("🎯 PromptFrame Library - Structure Verification")
    print("=" * 60)
    
    # Test imports
    try:
        from promptframe import PromptFrame, PromptFrameError
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return
    
    try:
        from promptframe.types import TemplateType, TemplateKwargsType
        print("✅ Type imports successful")
    except ImportError as e:
        print(f"❌ Type import failed: {e}")
        return
    
    try:
        from promptframe.utils import expand_pydantic_to_columns, merge_columns_to_dataframe
        print("✅ Utility imports successful")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return
    
    # Create sample schema
    class SimpleAnalysis(BaseModel):
        summary: str = Field(description="Brief summary")
        category: str = Field(description="Content category")
        
    print("✅ Pydantic schema creation successful")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        "text": [
            "This is a positive review of the product",
            "Negative feedback about service quality",
            "Neutral comment about the experience"
        ],
        "source": ["review", "feedback", "comment"]
    })
    
    print("✅ DataFrame creation successful")
    print(f"Sample data shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Test PromptFrame initialization
    try:
        pf = PromptFrame(df)
        print("✅ PromptFrame initialization successful")
        print(f"PromptFrame shape: {pf.shape}")
        print(f"PromptFrame columns: {list(pf.columns)}")
    except Exception as e:
        print(f"❌ PromptFrame initialization failed: {e}")
        return
    
    # Test utility functions
    try:
        sample_models = [
            SimpleAnalysis(summary="Test 1", category="positive"),
            SimpleAnalysis(summary="Test 2", category="negative")
        ]
        
        columns = expand_pydantic_to_columns(sample_models, "analysis")
        print("✅ Pydantic expansion successful")
        print(f"Generated columns: {list(columns.keys())}")
        
        merged_df = merge_columns_to_dataframe(df, columns)
        print("✅ DataFrame merging successful")
        print(f"Merged DataFrame shape: {merged_df.shape}")
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return
    
    # Test template rendering (without LLM calls)
    try:
        from promptframe.engine import AsyncEngine
        engine = AsyncEngine()
        
        # Test Jinja template
        test_row = df.iloc[0]
        template = "Text: {{ text }} | Source: {{ source }}"
        rendered = engine._render_prompt(template, test_row, None)
        print("✅ Jinja template rendering successful")
        print(f"Rendered template: {rendered}")
        
        # Test callable template
        def custom_template(row):
            return f"Custom: {row['text'][:20]}..."
        
        custom_rendered = engine._render_prompt(custom_template, test_row, None)
        print("✅ Callable template rendering successful")
        print(f"Custom rendered: {custom_rendered}")
        
    except Exception as e:
        print(f"❌ Template rendering test failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 All basic functionality tests passed!")
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the full demo: python src/promptframe/demo_quickstart.py")
    print("4. Run tests: pytest tests/")


if __name__ == "__main__":
    main()