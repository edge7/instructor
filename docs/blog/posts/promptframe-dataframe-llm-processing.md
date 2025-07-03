---
authors:
- jxnl
categories:
- Tools
- DataFrame Processing
comments: true
date: 2025-01-02
description: 'Introducing promptframe: A lightweight library for adding LLM-powered structured extraction to pandas DataFrames with automatic XML templating and async processing.'
draft: false
slug: promptframe-dataframe-llm-processing
tags:
- DataFrame
- Data Processing
- Structured Outputs
- Async
- XML Templates
- Pandas
---

# PromptFrame: DataFrame + LLM = Easy Structured Extraction

Working with tabular data and LLMs often involves repetitive boilerplate: iterating through rows, crafting prompts, making API calls, and parsing structured responses back into your DataFrame. What if this could be a one-liner?

Meet **promptframe** – a lightweight library that bridges pandas DataFrames and instructor-powered structured LLM extraction.

<!-- more -->

## The Problem

Data scientists and engineers frequently need to enrich tabular data with LLM insights:

- **Sentiment analysis** on customer reviews
- **Entity extraction** from text columns  
- **Classification** of support tickets
- **Summarization** of long text fields

The typical approach involves writing loops, managing API calls, handling errors, and manually adding results back to your DataFrame. It's tedious and error-prone.

## The Solution: PromptFrame

PromptFrame makes this trivial with automatic XML template generation and fast async processing:

```python
import pandas as pd
from pydantic import BaseModel
from promptframe import PromptFrame

# Define what you want to extract
class Sentiment(BaseModel):
    feeling: str      # positive, negative, neutral
    confidence: float # 0.0 to 1.0
    reason: str       # brief explanation

# Your data
df = pd.DataFrame({
    "review": [
        "I absolutely love this product!",
        "It's okay, nothing special.", 
        "Terrible quality, complete waste of money."
    ]
})

# Process with LLM (one line!)
result = (PromptFrame(df)
          .map_prompt("sentiment", Sentiment)
          .to_df())

print(result)
```

**Output:**
```
                           review sentiment.feeling  sentiment.confidence                   sentiment.reason
0        I absolutely love this product!       positive                  0.95  Enthusiastic positive language
1             It's okay, nothing special.        neutral                  0.80     Lukewarm, uncommitted tone
2  Terrible quality, complete waste of money.      negative                  0.98   Strong negative descriptors
```

## Key Features

### 🚀 **Zero-Setup Auto Templates**

No need to write prompts! PromptFrame automatically generates XML templates that wrap your DataFrame columns:

```xml
This is a row of a database, please extract the following information:
<review>{{ review }}</review>
```

Column names with spaces, hyphens, or dots are automatically sanitized for valid XML.

### ⚡ **Async Performance**

Process multiple rows concurrently with controlled concurrency:

```python
# Process 100 rows with up to 10 concurrent API calls
result = (PromptFrame(df)
          .map_prompt("analysis", AnalysisSchema, max_concurrency=10)
          .to_df())
```

**Performance comparison:**
- **Sequential**: 100 rows = ~5 minutes ⏳
- **Async**: 100 rows = ~30 seconds ⚡

**~10x faster processing!**

### 🔗 **Method Chaining**

Chain multiple LLM operations seamlessly:

```python
result = (PromptFrame(df)
          .map_prompt("sentiment", SentimentSchema)
          .map_prompt("topics", TopicSchema)  
          .map_prompt("summary", SummarySchema)
          .to_df())
```

### 🎯 **Custom Templates (Optional)**

When you need more control, provide custom templates:

```python
pf.map_prompt(
    "analysis", 
    AnalysisSchema,
    template="Rate this review on a scale of 1-10: {{ review }}"
)
```

## Real-World Example

Let's analyze customer support tickets:

```python
from pydantic import BaseModel, Field

class TicketAnalysis(BaseModel):
    priority: str = Field(description="low, medium, high, urgent")
    category: str = Field(description="technical, billing, general")
    sentiment: str = Field(description="frustrated, neutral, happy")
    needs_escalation: bool = Field(description="Should this be escalated?")

# Sample support tickets
tickets_df = pd.DataFrame({
    "ticket_text": [
        "My account was charged twice this month! This is unacceptable.",
        "How do I reset my password? Thanks for your help.",
        "The app keeps crashing when I try to upload files.",
        "Just wanted to say your customer service is amazing!",
        "URGENT: Our production server is down and losing money!"
    ],
    "customer_id": [1001, 1002, 1003, 1004, 1005]
})

# Analyze all tickets
analyzed = (PromptFrame(tickets_df)
            .map_prompt("analysis", TicketAnalysis, max_concurrency=5)
            .to_df())

# Now you can filter and prioritize
urgent_tickets = analyzed[analyzed['analysis.priority'] == 'urgent']
escalations = analyzed[analyzed['analysis.needs_escalation'] == True]
```

## Architecture: Simple by Design

The entire library is **120 lines of code** across 2 files:

```
promptframe/
├── src/promptframe/
│   ├── __init__.py              # 10 lines - Exports
│   └── promptframe.py           # 120 lines - Everything
├── requirements.txt             # 4 dependencies  
├── example.py                   # Demo
└── README.md                    # Usage guide
```

**Philosophy: Simple is better than complex.**

We originally built a complex version with 8 files, 746 lines, async engines, custom exceptions, and extensive error handling. Then we realized the core functionality could be distilled into something much simpler that's easier to understand, debug, and extend.

## Integration with Instructor

PromptFrame is built on top of `instructor`, inheriting all its benefits:

- **Automatic retries** with exponential backoff
- **Pydantic validation** with custom validators
- **Multiple provider support** (OpenAI, Anthropic, etc.)
- **Streaming support** for real-time processing

```python
# Use any instructor-compatible client
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())
pf = PromptFrame(df, client=client)
```

## Getting Started

Install and try it in 30 seconds:

```bash
pip install pandas pydantic instructor jinja2
export OPENAI_API_KEY=sk-...
```

```python
from promptframe import PromptFrame

# That's it! Start enriching your DataFrames
```

## When to Use PromptFrame

**Perfect for:**
- ✅ Adding LLM insights to tabular data
- ✅ Batch processing of text columns  
- ✅ Prototyping and analysis workflows
- ✅ ETL pipelines with structured extraction

**Consider alternatives for:**
- ❌ Single-row processing (just use instructor directly)
- ❌ Streaming large datasets (use chunking)
- ❌ Complex multi-step LLM workflows

## Try It Today

PromptFrame demonstrates the power of focused, simple tools that solve specific problems well. In just a few lines, you can add sophisticated LLM processing to any pandas workflow.

The complete example and source code are available in the [instructor examples](https://github.com/jxnl/instructor/tree/main/examples/promptframe).

Give it a try and let us know what you build! We'd love to see how you're using it to enhance your data processing workflows.

---

*PromptFrame is part of the instructor ecosystem. If you're working with structured LLM outputs, check out [instructor](https://python.useinstructor.com) for more advanced features and provider support.*