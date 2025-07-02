# PromptFrame - SIMPLIFIED ✨

## Before vs After

### Before: Complex 😵‍💫
- **8 Python files** (746 lines of code)
- **4 documentation files** 
- Async processing, semaphores, batching
- Custom exceptions, type definitions  
- Progress bars, error handling
- Complex templating system

### After: Simple 😊
- **1 Python file** (80 lines of code)
- **1 simple README**
- Synchronous, straightforward
- Basic error handling built-in
- Auto XML templates
- Just works

## Core Functionality (Unchanged)

```python
# Still does the same thing:
df = pd.DataFrame({"review": ["Great!", "Terrible."]})
result = (PromptFrame(df)
          .map_prompt("analysis", AnalysisSchema)
          .to_df())
```

## What We Removed

- ❌ Async/await complexity
- ❌ Custom exception classes  
- ❌ Progress bars
- ❌ Batching logic
- ❌ Complex type system
- ❌ Multiple template types
- ❌ Extensive error handling
- ❌ Development tooling

## What We Kept

- ✅ DataFrame + LLM processing
- ✅ Auto XML templates  
- ✅ Pydantic schemas
- ✅ Method chaining
- ✅ Column expansion
- ✅ Custom templates (optional)

## Files Now

```
promptframe/
├── src/promptframe/
│   ├── __init__.py          # 3 lines
│   └── promptframe.py       # 80 lines - EVERYTHING
├── simple_demo.py           # Example
├── requirements.txt         # 4 dependencies
└── README_SIMPLE.md         # Quick docs
```

## Usage (Exactly the Same)

```python
from promptframe import PromptFrame

# Same API, much simpler implementation
result = (PromptFrame(df)
          .map_prompt("sentiment", SentimentSchema)
          .to_df())
```

## Philosophy

> **"Simple is better than complex"** - The Zen of Python

The complex version had every feature imaginable. The simple version has the features you actually need.

Perfect for:
- ✅ Prototyping  
- ✅ Learning
- ✅ Simple projects
- ✅ Getting started

Not for:
- ❌ High-performance production systems
- ❌ Complex async workflows  
- ❌ Massive datasets
- ❌ Enterprise features

## Migration

If you were using the complex version:

```python
# Old (still works)
pf.map_prompt(
    name="analysis",
    template="Custom: {{ text }}",
    schema=Schema,
    batch_size=16,
    progress=True,
    temperature=0.7
)

# New (much simpler)
pf.map_prompt("analysis", Schema, "Custom: {{ text }}")
```

**Result: 90% fewer lines of code, same core functionality!**