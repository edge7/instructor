# promptframe

> Map Pandas rows → Jinja prompts → instructor-parsed JSON → new columns.

[![pypi](https://img.shields.io/pypi/v/promptframe)](https://pypi.org/project/promptframe)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

## ✨ Features
* **One-liner** to enrich DataFrames with LLM insights.
* **Structured outputs** validated by Pydantic via `instructor`.
* **Retry-safe** (Tenacity via Instructor) & **async by default**.
* Auto-expands nested fields (`analysis_summary`, …).

---

## 🔧 Installation

```bash
pip install promptframe
export OPENAI_API_KEY=sk-...
```

---

## 🚀 Quick Start

```python
import pandas as pd
from pydantic import BaseModel
from promptframe import PromptFrame

# 1. define schema
class Analysis(BaseModel):
    summary: str
    sentiment: str
    tags: list[str]

# 2. data
df = pd.DataFrame({"text": ["I love this.", "I hate that."]})

# 3. wrap & map
pf = PromptFrame(df)  # uses client=instructor.from_provider("openai/gpt-4o")

pf.map_prompt(
    name="analysis",
    template="""
      Text: {{ text }}
      
      Analyze this text and provide:
      - A brief summary
      - The sentiment (positive, negative, neutral)
      - Relevant tags
    """,
    schema=Analysis,
)

enriched = pf.to_df()
print(enriched.head())
```

**Output:**
```
        text analysis.summary analysis.sentiment analysis.tags
0  I love this.     Positive expression       positive    [love, positive]
1  I hate that.     Negative expression       negative    [hate, negative]
```

---

## 🛠 API In Detail

```python
PromptFrame(df, *, client=None, max_concurrency=32)

.map_prompt(
    name,                     # prefix for new columns
    template,                 # Jinja string or callable(row) -> str
    schema,                   # Pydantic model
    llm_model="openai/gpt-4o",
    template_kwargs=None,     # extra vars for Jinja
    batch_size=8,            # batch requests for efficiency
    progress=True,           # show progress bar
    **provider_kwargs,       # temperature, top_p, seed...
)
```

Columns are written as `{name}.{field}` for flat models, or dotted paths for nested models.

---

## 📋 Examples

### Advanced Template with Variables

```python
from pydantic import BaseModel, Field

class ProductAnalysis(BaseModel):
    category: str = Field(description="Product category")
    price_range: str = Field(description="Price range: low, medium, high")
    recommended: bool = Field(description="Whether to recommend this product")

# Add context via template_kwargs
pf.map_prompt(
    name="product_analysis",
    template="""
    Product: {{ name }}
    Description: {{ description }}
    Price: ${{ price }}
    
    Context: This analysis is for {{ target_audience }} customers 
    with a budget preference of {{ budget_pref }}.
    
    Analyze this product considering the target audience and budget.
    """,
    schema=ProductAnalysis,
    template_kwargs={
        "target_audience": "young professionals", 
        "budget_pref": "mid-range"
    }
)
```

### Custom Template Function

```python
def custom_template(row):
    return f"""
    Analyze this {row['content_type']}: {row['content']}
    
    Instructions:
    - Focus on {row['analysis_focus']}
    - Use {row['tone']} tone in response
    """

pf.map_prompt(
    name="custom_analysis",
    template=custom_template,
    schema=Analysis
)
```

### Batching for Efficiency

```python
# Process 16 rows at once (good for simple tasks)
pf.map_prompt(
    name="batch_analysis",
    template="Categorize this text: {{ text }}",
    schema=Category,
    batch_size=16  # Faster but less reliable for complex tasks
)
```

### Error Handling

```python
pf = PromptFrame(df)

try:
    pf.map_prompt(name="analysis", template="{{text}}", schema=Analysis)
except Exception as e:
    print(f"Processing failed: {e}")

# Check for partial errors
if pf.has_errors():
    print(f"Encountered {len(pf.get_errors())} errors")
    for error in pf.get_errors():
        print(f"- {error}")
```

---

## ⚙️ Configuration

| Env Var                       | Purpose                          |
| ----------------------------- | -------------------------------- |
| `OPENAI_API_KEY`              | required                         |
| `PROMPTFRAME_MAX_CONCURRENCY` | override default (32)            |
| `PROMPTFRAME_CACHE`           | `sqlite:///pf_cache.db` (future) |

---

## 🏗️ Advanced Usage

### Custom Instructor Client

```python
import instructor
from openai import AsyncOpenAI

# Use a different model or provider
client = instructor.from_provider("anthropic/claude-3-sonnet-20240229")
pf = PromptFrame(df, client=client)

# Or configure OpenAI with custom settings
openai_client = AsyncOpenAI(
    api_key="your-key",
    timeout=60.0,
)
client = instructor.from_provider(openai_client)
pf = PromptFrame(df, client=client)
```

### Chaining Operations

```python
(PromptFrame(df)
 .map_prompt("sentiment", sentiment_template, SentimentSchema)
 .map_prompt("topics", topic_template, TopicSchema, 
            template_kwargs=lambda row: {"sentiment": row["sentiment.label"]})
 .map_prompt("summary", summary_template, SummarySchema)
 .to_df())
```

### Progress Tracking

```python
# Custom progress tracking
pf.map_prompt(
    name="analysis",
    template=template,
    schema=Schema,
    progress=True,  # Shows tqdm progress bar
    batch_size=1   # More granular progress updates
)
```

---

## 🧑‍💻 Development

```bash
git clone your-org/promptframe
cd promptframe
pip install -e .
pip install -r requirements-dev.txt
pytest -v
```

Run the demo:

```bash
python src/promptframe/demo_quickstart.py
```

---

## 🚀 Performance Tips

1. **Batching**: Use `batch_size > 1` for simple, similar tasks
2. **Concurrency**: Adjust `max_concurrency` based on your API limits
3. **Templates**: Reuse template strings for automatic caching
4. **Progress**: Disable `progress=False` for faster processing in production

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT © 2025 Jason Liu & Contributors