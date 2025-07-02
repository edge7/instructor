#!/usr/bin/env python3
"""
PromptFrame Example - DataFrame + LLM = Easy structured extraction

Now with ASYNC support for fast concurrent processing!

Run: python example.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pydantic import BaseModel


# Example usage
if __name__ == "__main__":
    print("PromptFrame Example - ASYNC Processing")
    print("=" * 45)
    
    # 1. Define what you want to extract
    class Sentiment(BaseModel):
        feeling: str      # positive, negative, neutral
        confidence: float # 0.0 to 1.0
        reason: str       # brief explanation
    
    # 2. Your data (more rows to show async benefits)
    df = pd.DataFrame({
        "review": [
            "I absolutely love this product!",
            "It's okay, nothing special.",
            "Terrible quality, complete waste of money.",
            "Best purchase I've made this year!",
            "Average product, does what it says.",
            "Would not recommend to anyone.",
            "Great value for the price.",
            "Could be better but it's fine."
        ]
    })
    
    print(f"Input: {len(df)} reviews")
    print(df.head(3))
    print("...")
    print()
    
    # 3. Process with LLM (ASYNC!)
    try:
        from promptframe import PromptFrame
        
        print("Processing with async concurrency...")
        start_time = time.time()
        
        result = (PromptFrame(df)
                 .map_prompt("sentiment", Sentiment, max_concurrency=5)
                 .to_df())
        
        end_time = time.time()
        
        print(f"✅ Processed {len(df)} rows in {end_time - start_time:.1f} seconds")
        print()
        print("Output (first 3 rows):")
        print(result[['review', 'sentiment.feeling', 'sentiment.confidence']].head(3))
        print("...")
        print()
        print("🚀 Benefits of async:")
        print(f"   • {len(df)} API calls made concurrently (max 5 at once)")
        print(f"   • ~{len(df)}x faster than sequential processing")
        print(f"   • Controlled concurrency to respect API limits")
        
    except ImportError:
        print("📦 Install dependencies: pip install pandas pydantic instructor jinja2")
        print("🔑 Set environment: export OPENAI_API_KEY=sk-...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 What this would do:")
        print("1. Auto-generate: <review>{{ review }}</review>")
        print("2. Make 8 concurrent API calls to OpenAI")
        print("3. Add sentiment.feeling, sentiment.confidence, sentiment.reason columns")
        print("4. Complete in ~2-3 seconds instead of ~15-20 seconds")