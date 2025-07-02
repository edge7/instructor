# promptframe

> Map Pandas rows → Jinja prompts → instructor-parsed JSON → new columns.

[![pypi](https://img.shields.io/pypi/v/promptframe)](https://pypi.org/project/promptframe)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

## ✨ Features
* **One-liner** to enrich DataFrames with LLM insights.
* **Structured outputs** validated by Pydantic via `instructor`.
* **Retry-safe** (Tenacity via Instructor) & **async by default**.
* **Default XML templates** - No need to write templates for simple extraction!
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

# 3. wrap & map (NO TEMPLATE NEEDED!)
pf = PromptFrame(df)

pf.map_prompt(
    name="analysis",
    schema=Analysis,
    # No template parameter - uses automatic XML template!
)

enriched = pf.to_df()
print(enriched.head())
```

**Auto-generated template:**
```xml
This is a row of a database, please extract the following information:
<text>{{ text }}</text>
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
    schema,                   # Pydantic model
    template=None,           # Jinja string, callable, or None (auto XML)
    llm_model="openai/gpt-4o",
    template_kwargs=None,     # extra vars for Jinja
    batch_size=8,            # batch requests for efficiency
    progress=True,           # show progress bar
    **provider_kwargs,       # temperature, top_p, seed...
)
```

**Template Options:**
- `None` (default): Auto-generates XML template wrapping each column
- `str`: Custom Jinja2 template
- `callable`: Function that takes a row and returns a string

Columns are written as `{name}.{field}` for flat models, or dotted paths for nested models.

---

## 📋 Examples

### No Template Needed (Default XML)

```python
from pydantic import BaseModel

class PersonInfo(BaseModel):
    full_name: str
    age_group: str  # child, teen, adult, senior
    profession: str

df = pd.DataFrame({
    "name": ["Alice Johnson", "Bob Smith"],
    "age": [28, 45],
    "job": ["Engineer", "Manager"],
    "city": ["SF", "NYC"]
})

# Just specify name and schema - template auto-generated!
pf = PromptFrame(df)
pf.map_prompt(name="person", schema=PersonInfo)

# Auto-generated template:
# This is a row of a database, please extract the following information:
# <name>{{ name }}</name>
# <age>{{ age }}</age>
# <job>{{ job }}</job>
# <city>{{ city }}</city>
```

### Custom Template (When You Need More Control)

```python
from pydantic import BaseModel, Field

class ProductAnalysis(BaseModel):
    category: str = Field(description="Product category")
    price_range: str = Field(description="Price range: low, medium, high")
    recommended: bool = Field(description="Whether to recommend this product")

# Custom template with context
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

### Column Name Sanitization

```python
# DataFrame with challenging column names
df = pd.DataFrame({
    "user name": ["Alice", "Bob"],           # spaces
    "email-address": ["a@test.com", "b@test.com"],  # hyphens  
    "signup.date": ["2023-01-15", "2023-02-20"],    # dots
})

# Auto-generated template sanitizes column names:
# <user_name>{{ user name }}</user_name>
# <email_address>{{ email-address }}</email_address>  
# <signup_date>{{ signup.date }}</signup_date>
```

### Batching for Efficiency

```python
# Process 16 rows at once (good for simple tasks)
pf.map_prompt(
    name="batch_analysis",
    schema=Category,
    batch_size=16  # Faster but less reliable for complex tasks
)
```

### Error Handling

```python
pf = PromptFrame(df)

try:
    pf.map_prompt(name="analysis", schema=Analysis)
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
 .map_prompt("sentiment", SentimentSchema)  # No template needed
 .map_prompt("topics", TopicSchema, 
            template_kwargs=lambda row: {"sentiment": row["sentiment.label"]})
 .map_prompt("summary", SummarySchema)
 .to_df())
```

### Template Comparison

```python
# Method 1: Default XML template (zero-setup)
pf1 = PromptFrame(df)
pf1.map_prompt(name="auto", schema=Schema)

# Method 2: Custom template (full control)  
pf2 = PromptFrame(df)
pf2.map_prompt(
    name="custom", 
    template="Custom prompt: {{ column }}",
    schema=Schema
)
```

---

## 🧑‍💻 Development

```bash
git clone your-org/promptframe
cd promptframe
pip install -e .
pip install -r requirements.txt
pytest -v
```

Run the demos:

```bash
python src/promptframe/demo_quickstart.py          # Original demos
python demo_default_template.py                    # New default template demos
```

---

## 🚀 Performance Tips

1. **Default templates**: Use auto-generated XML templates for quick prototyping
2. **Batching**: Use `batch_size > 1` for simple, similar tasks
3. **Concurrency**: Adjust `max_concurrency` based on your API limits
4. **Templates**: Reuse template strings for automatic caching
5. **Progress**: Disable `progress=False` for faster processing in production

---

## 💡 When to Use Each Template Type

| Template Type | Best For | Example |
|---------------|----------|---------|
| **None (XML)** | Quick extraction, prototyping | `map_prompt("data", Schema)` |
| **Custom String** | Specific formatting, context | Complex prompts with instructions |
| **Callable** | Dynamic prompts, row-dependent logic | Different prompts per content type |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT © 2025 Jason Liu & Contributors