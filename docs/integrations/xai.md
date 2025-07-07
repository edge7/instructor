# xAI Provider

The xAI provider allows you to use [xAI's Grok models](https://x.ai/) with Instructor for structured outputs. xAI provides an OpenAI-compatible API, making integration seamless.

## Installation

Since xAI uses an OpenAI-compatible API, you'll need to install the OpenAI SDK:

```bash
pip install openai instructor
```

## Setup

Get your xAI API key from [console.x.ai](https://console.x.ai) and set it as an environment variable:

```bash
export XAI_API_KEY="your-api-key-here"
```

## Basic Usage

### Using `from_xai`

```python
import instructor
from openai import OpenAI

# Create xAI client
client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# Patch with instructor
instructor_client = instructor.from_xai(client)
```

### Using `from_provider`

The easiest way to get started is using the `from_provider` method:

```python
import instructor

# Automatically creates the xAI client
client = instructor.from_provider("xai/grok-beta")
```

## Modes

The xAI provider supports two modes:

1. **`XAI_TOOLS`** (default): Uses function calling for structured outputs
2. **`XAI_JSON`**: Uses JSON mode for simpler structured outputs

### Function Calling Mode (Tools)

```python
from pydantic import BaseModel, Field
import instructor

class User(BaseModel):
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")

client = instructor.from_provider("xai/grok-beta")

user = client.chat.completions.create(
    model="grok-beta",
    messages=[
        {"role": "user", "content": "Extract: John Doe's email is john@example.com"}
    ],
    response_model=User,
)

print(user.name)   # "John Doe"
print(user.email)  # "john@example.com"
```

### JSON Mode

```python
client = instructor.from_provider(
    "xai/grok-beta", 
    mode=instructor.Mode.XAI_JSON
)

user = client.chat.completions.create(
    model="grok-beta",
    messages=[
        {"role": "user", "content": "Extract: John Doe's email is john@example.com"}
    ],
    response_model=User,
)
```

## Available Models

As of January 2025, xAI offers:

- `grok-beta` - The latest Grok model
- `grok-3-beta` - Grok 3 beta version
- `grok-3-mini-beta` - A smaller, faster version of Grok 3

Check the [xAI documentation](https://docs.x.ai/) for the latest available models.

## Advanced Usage

### Complex Nested Structures

```python
from typing import List
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

client = instructor.from_provider("xai/grok-beta")

person = client.chat.completions.create(
    model="grok-beta",
    messages=[
        {
            "role": "user",
            "content": """
            John Smith is 30 years old. He lives at 123 Main St, 
            New York, USA and has a vacation home at 456 Beach Rd, 
            Miami, USA.
            """
        }
    ],
    response_model=Person,
)

print(f"{person.name} has {len(person.addresses)} addresses")
```

### Async Support

```python
client = instructor.from_provider("xai/grok-beta", async_client=True)

user = await client.chat.completions.create(
    model="grok-beta",
    messages=[{"role": "user", "content": "Extract user info..."}],
    response_model=User,
)
```

## Error Handling

The xAI provider includes built-in error handling:

```python
from instructor.exceptions import InstructorRetryException

try:
    result = client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": "Extract data..."}],
        response_model=MyModel,
        max_retries=3,
    )
except InstructorRetryException as e:
    print(f"Failed after retries: {e}")
```

## Best Practices

1. **Choose the Right Mode**: Use `XAI_TOOLS` for complex schemas and `XAI_JSON` for simpler ones
2. **API Key Security**: Always use environment variables for API keys
3. **Model Selection**: Choose `grok-3-mini-beta` for faster responses when accuracy requirements are lower
4. **Rate Limits**: Be aware of xAI's rate limits and implement appropriate retry logic

## Limitations

- Streaming is not yet implemented in this provider
- xAI's API is in beta and features may change
- Function calling capabilities depend on the specific Grok model used

## See Also

- [xAI Documentation](https://docs.x.ai/)
- [Instructor Documentation](https://python.useinstructor.com/)
- [Example Code](https://github.com/jxnl/instructor/blob/main/examples/xai_example.py)