# PromptFrame - Quick Start Guide

This directory contains the complete **promptframe** library implementation with **automatic XML template generation**.

## 📁 Directory Structure

```
examples/promptframe/
├── src/promptframe/          # Main library code
│   ├── __init__.py          # Package entry point
│   ├── core.py              # PromptFrame class
│   ├── engine.py            # Async LLM processing engine
│   ├── types.py             # Type definitions and exceptions
│   ├── utils.py             # Utility functions
│   └── demo_quickstart.py   # Full demo script
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_promptframe.py
├── demo_default_template.py # NEW: Default XML template demos
├── DESIGN_DOC.md           # Technical specification
├── README.md               # Full documentation
├── requirements.txt        # Dependencies
├── pyproject.toml         # Package configuration
├── Makefile               # Development commands
├── simple_example.py      # Basic verification script
└── QUICKSTART.md          # This file
```

## 🚀 Getting Started

### 1. Quick Verification (No API Required)

Test the library structure without making API calls:

```bash
cd examples/promptframe
python simple_example.py
```

This will verify all imports and basic functionality work correctly.

### 2. Install Dependencies

```bash
# Install the required packages
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### 3. Set API Key

```bash
export OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 4. Run Demos

```bash
# Original comprehensive demos
python src/promptframe/demo_quickstart.py

# NEW: Default XML template demos
python demo_default_template.py
```

## 💡 Basic Usage (NEW: No Template Required!)

```python
import pandas as pd
from pydantic import BaseModel
from promptframe import PromptFrame

# Define your schema
class Analysis(BaseModel):
    summary: str
    sentiment: str
    tags: list[str]

# Create your data
df = pd.DataFrame({"text": ["I love this!", "This is terrible."]})

# Process with LLM - NO TEMPLATE NEEDED!
pf = PromptFrame(df)
pf.map_prompt(
    name="analysis",
    schema=Analysis
    # template parameter is optional - auto-generates XML template!
)

# Get enriched DataFrame
result = pf.to_df()
print(result)
```

**What happens automatically:**
1. PromptFrame sees you didn't provide a `template`
2. It auto-generates an XML template wrapping each column:
   ```xml
   This is a row of a database, please extract the following information:
   <text>{{ text }}</text>
   ```
3. This template is used for all rows in your DataFrame

## 🎯 Key New Feature: Default XML Templates

### ✨ Zero Setup Required
```python
# OLD way (still works)
pf.map_prompt(
    name="analysis",
    template="Analyze: {{ text }}",  # Had to write this
    schema=Analysis
)

# NEW way (zero setup!)
pf.map_prompt(
    name="analysis", 
    schema=Analysis  # Template auto-generated!
)
```

### 🧹 Automatic Column Sanitization
```python
# DataFrame with messy column names
df = pd.DataFrame({
    "user name": ["Alice"],          # spaces
    "email-address": ["a@test.com"], # hyphens
    "signup.date": ["2023-01-15"]    # dots
})

# Auto-generated template handles this:
# <user_name>{{ user name }}</user_name>
# <email_address>{{ email-address }}</email_address>  
# <signup_date>{{ signup.date }}</signup_date>
```

### 🔄 Template Flexibility
```python
# Method 1: Automatic (great for prototyping)
pf.map_prompt(name="auto", schema=Schema)

# Method 2: Custom string (when you need control)
pf.map_prompt(
    name="custom",
    template="Special instructions: {{ column }}",
    schema=Schema
)

# Method 3: Dynamic function (for complex logic)
def my_template(row):
    if row['type'] == 'urgent':
        return f"URGENT: {row['message']}"
    return f"Normal: {row['message']}"

pf.map_prompt(name="dynamic", template=my_template, schema=Schema)
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run tests without coverage
make test-fast

# Run specific test file
pytest tests/test_promptframe.py -v
```

## 🛠 Development

```bash
# Format code
make format

# Lint code  
make lint

# Run all checks
make check

# Clean build artifacts
make clean
```

## 🏗 Key Features Implemented

### ✅ Core API
- **PromptFrame class** - Main interface wrapping DataFrames
- **map_prompt method** - Applies LLM to rows with structured output
- **🆕 Default XML templates** - Zero-setup automatic template generation
- **Method chaining** - Fluent interface for multiple operations
- **Error handling** - Graceful error capture and reporting

### ✅ Template System  
- **🆕 Automatic XML templates** - No template writing required
- **🆕 Column sanitization** - Handles spaces, hyphens, dots in column names
- **Jinja2 templates** - String-based templates with variable substitution
- **Callable templates** - Python functions for dynamic prompt generation
- **Template caching** - Automatic caching of compiled templates
- **Context variables** - Support for additional template parameters

### ✅ Async Processing
- **Concurrency control** - Configurable semaphore limits
- **Batch processing** - Optional batching for efficiency
- **Progress tracking** - tqdm integration for long-running operations
- **Error isolation** - Continue processing when individual calls fail

### ✅ Data Integration
- **Column expansion** - Automatic flattening of nested Pydantic models
- **Type safety** - Full type hints throughout
- **DataFrame preservation** - Non-destructive operations on original data
- **Flexible schemas** - Any Pydantic BaseModel supported

## 📋 Updated API

### Core Class (unchanged)
```python
PromptFrame(df, *, client=None, max_concurrency=32)
```

### Main Method (template now optional!)
```python
.map_prompt(
    name: str,                    # Column prefix
    schema: Type[BaseModel],      # Pydantic schema  
    template: Optional[str | Callable] = None,  # 🆕 Now optional!
    *,
    llm_model="openai/gpt-4o",   # Model to use
    template_kwargs=None,         # Extra template variables
    batch_size=8,                # Batch size for efficiency
    progress=True,               # Show progress bar
    **provider_kwargs            # LLM provider arguments
) -> PromptFrame
```

**🆕 Template Parameter:**
- `None` (default): Auto-generates XML template wrapping each column
- `str`: Custom Jinja2 template  
- `callable`: Function taking a row, returning a string

## 🔧 Configuration

Environment variables:
- `OPENAI_API_KEY` - Required for OpenAI API access
- `PROMPTFRAME_MAX_CONCURRENCY` - Override default concurrency (32)

## 📖 Next Steps

1. **Try the new defaults**: Use `map_prompt()` without a template!
2. **Run the demos**: `python demo_default_template.py`
3. **Read the full docs**: [README.md](README.md) for all examples
4. **Check technical details**: [DESIGN_DOC.md](DESIGN_DOC.md)
5. **Explore tests**: `tests/test_promptframe.py` shows API usage

## 🎉 What's New

### Before (v0.1.0)
```python
# Always required a template
pf.map_prompt(
    name="analysis",
    template="Text: {{ text }}\nAnalyze this.",  # Required!
    schema=Schema
)
```

### After (v0.1.1)  
```python
# Template is optional - auto-generates XML
pf.map_prompt(
    name="analysis", 
    schema=Schema  # That's it!
)

# Still supports custom templates when needed
pf.map_prompt(
    name="custom",
    template="Special: {{ text }}",
    schema=Schema
)
```

## 🤝 Contributing

This is an example implementation. For production use:
1. Add more comprehensive error handling
2. Implement caching mechanisms  
3. Add support for more LLM providers
4. Optimize batching strategies
5. Add streaming support for large datasets

## 📜 License

MIT © 2025 - Built as an example for the instructor library ecosystem.

**🆕 Features added:**
- ✅ Default XML template generation
- ✅ Column name sanitization  
- ✅ Optional template parameter
- ✅ Comprehensive demos and documentation