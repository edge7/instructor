# promptframe

DataFrame + LLM = Easy structured extraction + **FAST async processing**

## Install & Run

```bash
pip install pandas pydantic instructor jinja2
export OPENAI_API_KEY=sk-...
python example.py
```

## Usage

```python
from promptframe import PromptFrame
from pydantic import BaseModel

class Sentiment(BaseModel):
    feeling: str      # positive, negative, neutral
    confidence: float # 0.0 to 1.0

df = pd.DataFrame({"review": ["Great product!", "Terrible."]})

# Process with async concurrency (FAST!)
result = (PromptFrame(df)
          .map_prompt("sentiment", Sentiment, max_concurrency=10)
          .to_df())

print(result)
```

**Output:**
```
       review sentiment.feeling  sentiment.confidence
0  Great product!      positive                  0.95
1      Terrible.      negative                  0.90
```

## ⚡ Async Performance

- **Sequential**: 10 rows = 10 API calls = ~30 seconds
- **Async**: 10 rows = 10 concurrent calls = ~3 seconds

**~10x faster processing!**

## Files

- `src/promptframe/promptframe.py` - Everything (120 lines)
- `example.py` - Demo with 8 rows showing async speed

## How it works

1. Auto-generates XML template: `<review>{{ review }}</review>`
2. Makes **concurrent** API calls to OpenAI (controlled by `max_concurrency`)
3. Adds new columns with structured results

## API

```python
.map_prompt(name, schema, template=None, max_concurrency=10)
```

- `max_concurrency=10` - Up to 10 API calls at once
- Higher = faster, but respect API rate limits
- Default 10 is safe for most use cases

Done.