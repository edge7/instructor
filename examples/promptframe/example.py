#!/usr/bin/env python3
"""
PromptFrame Example - DataFrame + LLM = Easy structured extraction

Run: python example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pydantic import BaseModel


# Example usage
if __name__ == "__main__":
    print("PromptFrame Example")
    print("=" * 30)
    
    # 1. Define what you want to extract
    class Sentiment(BaseModel):
        feeling: str      # positive, negative, neutral
        confidence: float # 0.0 to 1.0
    
    # 2. Your data
    df = pd.DataFrame({
        "review": [
            "I absolutely love this product!",
            "It's okay, nothing special.",
            "Terrible quality, complete waste of money."
        ]
    })
    
    print("Input:")
    print(df)
    print()
    
    # 3. Process with LLM
    try:
        from promptframe import PromptFrame
        
        result = (PromptFrame(df)
                 .map_prompt("sentiment", Sentiment)
                 .to_df())
        
        print("Output:")
        print(result[['review', 'sentiment.feeling', 'sentiment.confidence']])
        
    except ImportError as e:
        print(f"Install dependencies: pip install pandas pydantic instructor jinja2")
        print(f"Set environment: export OPENAI_API_KEY=sk-...")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nWhat this would do:")
        print("1. Auto-generate: <review>{{ review }}</review>")
        print("2. Send to OpenAI with Sentiment schema")
        print("3. Add sentiment.feeling and sentiment.confidence columns")