#!/usr/bin/env python3
"""
Super simple PromptFrame demo - One file, minimal complexity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pydantic import BaseModel
from promptframe import PromptFrame


class Sentiment(BaseModel):
    feeling: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0


def main():
    print("🚀 PromptFrame - Super Simple Demo")
    print("=" * 40)
    
    # 1. Your data
    df = pd.DataFrame({
        "review": [
            "I absolutely love this product!",
            "Meh, it's okay I guess.",
            "Terrible quality, waste of money."
        ]
    })
    
    print("Input data:")
    print(df)
    print()
    
    # 2. Process with LLM (one line!)
    try:
        result = (PromptFrame(df)
                 .map_prompt("sentiment", Sentiment)
                 .to_df())
        
        print("Results:")
        print(result[['review', 'sentiment.feeling', 'sentiment.confidence']])
        
    except Exception as e:
        print(f"Demo requires OPENAI_API_KEY: {e}")
        print("\nWhat WOULD happen:")
        print("- Auto-generates: <review>{{ review }}</review>")
        print("- Sends to OpenAI with your Sentiment schema")
        print("- Adds sentiment.feeling and sentiment.confidence columns")


if __name__ == "__main__":
    main()