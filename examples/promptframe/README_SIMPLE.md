# promptframe

> DataFrame + LLM = Easy structured extraction

**One file. Three dependencies. Zero complexity.**

## Install

```bash
pip install pandas pydantic instructor
export OPENAI_API_KEY=sk-...
```

## Use

```python
import pandas as pd
from pydantic import BaseModel
from promptframe import PromptFrame

# Define what you want to extract
class Analysis(BaseModel):
    sentiment: str
    summary: str

# Your data
df = pd.DataFrame({"review": ["Great product!", "Terrible service."]})

# Process with LLM
result = (PromptFrame(df)
          .map_prompt("analysis", Analysis)
          .to_df())

print(result)
```

**Output:**
```
       review analysis.sentiment  analysis.summary
0  Great product!        positive   Customer loves it
1  Terrible service.     negative   Poor service quality
```

## That's it!

### What happens automatically:
1. **Auto-template**: Wraps your columns in XML like `<review>{{ review }}</review>`
2. **LLM call**: Sends to OpenAI with your Pydantic schema
3. **Add columns**: Results become new DataFrame columns with dotted names

### Custom template (optional):
```python
pf.map_prompt("analysis", Analysis, template="Rate this review: {{ review }}")
```

### Chain operations:
```python
result = (PromptFrame(df)
          .map_prompt("sentiment", SentimentSchema)
          .map_prompt("topics", TopicSchema)
          .to_df())
```

## Files

- `src/promptframe/promptframe.py` - Everything (80 lines)
- `src/promptframe/__init__.py` - Exports
- `simple_demo.py` - Example

## Dependencies

- `pandas` - DataFrames
- `pydantic` - Schemas  
- `instructor` - Structured LLM calls
- `jinja2` - Templates

Done.