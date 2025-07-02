# PromptFrame - Quick Start Guide

This directory contains the complete **promptframe** library implementation based on the design specifications.

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

### 4. Run Full Demo

```bash
python src/promptframe/demo_quickstart.py
```

This demonstrates:
- Basic text analysis with structured output
- Advanced templates with variables
- Custom template functions
- Progress tracking and error handling

## 💡 Basic Usage

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

# Process with LLM
pf = PromptFrame(df)
pf.map_prompt(
    name="analysis",
    template="Analyze this text: {{ text }}",
    schema=Analysis
)

# Get enriched DataFrame
result = pf.to_df()
print(result)
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

- ✅ **PromptFrame class** - Main interface wrapping DataFrames
- ✅ **Async engine** - Concurrent LLM processing with semaphore control
- ✅ **Template support** - Both Jinja2 strings and callable functions
- ✅ **Column expansion** - Automatic flattening of nested Pydantic models
- ✅ **Error handling** - Graceful error capture and reporting
- ✅ **Progress tracking** - Optional tqdm progress bars
- ✅ **Batching support** - Configurable batch sizes for efficiency
- ✅ **Type safety** - Full type hints and Pydantic validation

## 📋 API Overview

### Core Class

```python
PromptFrame(df, *, client=None, max_concurrency=32)
```

### Main Method

```python
.map_prompt(
    name: str,                    # Column prefix
    template: str | Callable,     # Jinja template or function
    schema: Type[BaseModel],      # Pydantic schema
    *,
    llm_model="openai/gpt-4o",   # Model to use
    template_kwargs=None,         # Extra template variables
    batch_size=8,                # Batch size for efficiency
    progress=True,               # Show progress bar
    **provider_kwargs            # LLM provider arguments
) -> PromptFrame
```

## 🔧 Configuration

Environment variables:
- `OPENAI_API_KEY` - Required for OpenAI API access
- `PROMPTFRAME_MAX_CONCURRENCY` - Override default concurrency (32)

## 📖 Next Steps

1. Read the full [README.md](README.md) for detailed examples
2. Check the [DESIGN_DOC.md](DESIGN_DOC.md) for technical details
3. Explore the demo script: `src/promptframe/demo_quickstart.py`
4. Run tests to understand the API: `tests/test_promptframe.py`

## 🤝 Contributing

This is an example implementation. For production use:
1. Add more comprehensive error handling
2. Implement caching mechanisms
3. Add support for more LLM providers
4. Optimize batching strategies
5. Add streaming support for large datasets

## 📜 License

MIT © 2025 - Built as an example for the instructor library ecosystem.