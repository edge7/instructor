# promptframe

DataFrame + LLM = Easy structured extraction

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

result = (PromptFrame(df)
          .map_prompt("sentiment", Sentiment)
          .to_df())

print(result)
```

**Output:**
```
       review sentiment.feeling  sentiment.confidence
0  Great product!      positive                  0.95
1      Terrible.      negative                  0.90
```

## Files

- `src/promptframe/promptframe.py` - Everything (110 lines)
- `example.py` - Demo

## How it works

1. Auto-generates XML template: `<review>{{ review }}</review>`
2. Sends to OpenAI with your Pydantic schema
3. Adds new columns with structured results

Done.