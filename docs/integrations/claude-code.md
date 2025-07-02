---
title: "Claude Code CLI Integration: Structured Outputs with Instructor"
description: "Complete guide to using Claude Code CLI with Instructor for structured data extraction without API keys. Learn how to leverage the Claude Code CLI for type-safe outputs in Python."
---

# Claude Code CLI Integration: Structured Outputs with Instructor

Learn how to use Claude Code CLI with Instructor to extract structured, validated data without managing API keys. This integration lets you use the Claude Code CLI directly from your Python applications while getting the benefits of structured outputs.

## Quick Start: Install Claude Code CLI

First, ensure you have Claude Code CLI installed and working:

```bash
# Install Claude Code CLI (follow official installation instructions)
# Verify installation
claude --version
```

Then install Instructor:

```bash
pip install instructor
```

### Basic Usage

```python
# Standard library imports
from typing import List

# Third-party imports
import instructor
from pydantic import BaseModel, Field

# Define your models with proper type annotations
class Properties(BaseModel):
    """Model representing a key-value property."""
    name: str = Field(description="The name of the property")
    value: str = Field(description="The value of the property")


class User(BaseModel):
    """Model representing a user with properties."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    properties: List[Properties] = Field(description="List of user properties")

# Initialize the Claude Code client
client = instructor.from_claude_code()

# Extract structured data
response = client.create(
    response_model=User,
    messages=[
        {
            "role": "user", 
            "content": "Extract information about Jason: He's 25 years old and works as a software engineer."
        }
    ],
)

print(response)
# User(name='Jason', age=25, properties=[Properties(name='occupation', value='software engineer')])
```

## Installation and Setup

### Prerequisites

1. **Claude Code CLI**: Install and configure Claude Code CLI following the official documentation
2. **Python Environment**: Python 3.8 or higher
3. **Instructor**: Install with `pip install instructor`

### Verification

Test that everything is working:

```python
import subprocess

# Check if Claude CLI is available
try:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    print("Claude CLI is available:", result.returncode == 0)
except FileNotFoundError:
    print("Claude CLI not found in PATH")
```

## Configuration Options

### Basic Configuration

```python
import instructor

# Default configuration (uses "claude" command)
client = instructor.from_claude_code()

# Custom CLI path
client = instructor.from_claude_code(cli_path="/usr/local/bin/claude")

# Specify model
client = instructor.from_claude_code(model="claude-3-5-sonnet-20241022")

# Async client
async_client = instructor.from_claude_code(async_client=True)
```

### Advanced Configuration

```python
# With additional CLI parameters
client = instructor.from_claude_code(
    model="claude-3-5-sonnet",
    max_tokens=4000,
    temperature=0.1,
    timeout=120  # Custom timeout in seconds
)
```

## Supported Modes

Claude Code integration uses JSON mode for structured outputs:

- **CLAUDE_CODE_JSON**: Uses JSON schema instructions to guide Claude Code CLI

## Usage Patterns

### Simple Data Extraction

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = instructor.from_claude_code()

person = client.create(
    response_model=Person,
    messages=[
        {"role": "user", "content": "Sarah is a 28-year-old doctor."}
    ]
)

print(person.name)  # Sarah
print(person.age)   # 28
print(person.occupation)  # doctor
```

### Complex Nested Models

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    name: str = Field(description="Full name")
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None

class Company(BaseModel):
    name: str
    employees: List[Contact]
    headquarters: Address

client = instructor.from_claude_code()

company = client.create(
    response_model=Company,
    messages=[
        {
            "role": "user",
            "content": """
            Acme Corp is headquartered at 123 Main St, Springfield, IL 62701.
            The company has employees John Doe (john@acme.com) and Jane Smith (jane@acme.com, 555-1234).
            """
        }
    ]
)

print(company.name)  # Acme Corp
print(len(company.employees))  # 2
print(company.headquarters.city)  # Springfield
```

### Async Usage

```python
import asyncio
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: List[str]
    conclusion: str

async def extract_summary():
    client = instructor.from_claude_code(async_client=True)
    
    summary = await client.create(
        response_model=Summary,
        messages=[
            {
                "role": "user",
                "content": "Summarize this article: [long article text here]"
            }
        ]
    )
    
    return summary

# Run async function
summary = asyncio.run(extract_summary())
print(summary.title)
```

## Provider Integration

Use Claude Code through the unified provider interface:

```python
import instructor

# Via provider interface
client = instructor.from_provider("claude_code/claude-3-5-sonnet")

# With custom CLI path
client = instructor.from_provider(
    "claude_code/claude-3-5-sonnet",
    cli_path="/usr/local/bin/claude"
)
```

## Error Handling

### CLI Not Available

```python
import instructor

try:
    client = instructor.from_claude_code()
except RuntimeError as e:
    print(f"Claude CLI error: {e}")
    # Handle the error - perhaps fallback to API-based provider
```

### Timeout Handling

```python
try:
    response = client.create(
        response_model=MyModel,
        messages=messages,
        timeout=60  # Custom timeout
    )
except RuntimeError as e:
    if "timeout" in str(e):
        print("Request timed out")
    else:
        print(f"Other error: {e}")
```

### JSON Parsing Errors

```python
from pydantic import ValidationError

try:
    response = client.create(
        response_model=MyModel,
        messages=messages
    )
except ValueError as e:
    if "Failed to parse response as JSON" in str(e):
        print("Claude CLI returned invalid JSON")
        print(f"Error: {e}")
except ValidationError as e:
    print(f"Response doesn't match model: {e}")
```

## Best Practices

### 1. Model Design

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class WellDesignedModel(BaseModel):
    """Always include class docstrings for better results."""
    
    required_field: str = Field(description="Clear field descriptions help")
    optional_field: Optional[str] = Field(None, description="Make optional fields explicit")
    list_field: List[str] = Field(description="Describe list contents")
    
    class Config:
        # Enable validation on assignment
        validate_assignment = True
```

### 2. Message Structure

```python
# Clear, specific instructions work best
messages = [
    {
        "role": "user",
        "content": "Extract company information from this text. Include name, industry, and employee count: [text here]"
    }
]

# For complex tasks, use system messages if needed
messages = [
    {
        "role": "system",
        "content": "You are an expert at extracting structured data from unstructured text."
    },
    {
        "role": "user",
        "content": "Extract the requested information: [text here]"
    }
]
```

### 3. Error Recovery

```python
def robust_extraction(client, text, model_class, max_retries=3):
    """Extract data with retry logic."""
    for attempt in range(max_retries):
        try:
            return client.create(
                response_model=model_class,
                messages=[{"role": "user", "content": text}]
            )
        except (ValueError, RuntimeError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
```

## Comparison with API-based Providers

| Feature | Claude Code CLI | Anthropic API |
|---------|-----------------|---------------|
| Authentication | Handled by CLI | Requires API key |
| Rate Limiting | CLI managed | Manual handling |
| Cost | Per CLI usage | Per API call |
| Latency | Subprocess overhead | Direct HTTP |
| Offline Usage | Possible* | No |
| Configuration | CLI config | Code config |

*Depending on Claude Code CLI capabilities

## Troubleshooting

### Common Issues

1. **CLI Not Found**
   ```
   RuntimeError: Claude CLI not found at 'claude'
   ```
   Solution: Ensure Claude Code CLI is installed and in PATH

2. **Permission Denied**
   ```
   PermissionError: [Errno 13] Permission denied: 'claude'
   ```
   Solution: Check file permissions and execution rights

3. **Timeout Errors**
   ```
   RuntimeError: Claude CLI request timed out
   ```
   Solution: Increase timeout or check system resources

4. **JSON Parsing Errors**
   ```
   ValueError: Failed to parse response as JSON
   ```
   Solution: Check model complexity or message clarity

### Debug Mode

Enable verbose logging to debug issues:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
client = instructor.from_claude_code()
```

## Advanced Features

### Custom CLI Arguments

```python
# Pass additional arguments to Claude CLI
client = instructor.from_claude_code(
    max_tokens=8000,
    temperature=0.2,
    custom_arg="value"  # Passed as --custom-arg value
)
```

### Response Validation

```python
from pydantic import validator

class ValidatedModel(BaseModel):
    email: str
    age: int
    
    @validator('email')
    def validate_email(cls, v):
        # Add custom validation
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Invalid age')
        return v
```

### Streaming Support

```python
# Note: Streaming support depends on Claude Code CLI capabilities
# Implementation may vary based on CLI version
async def stream_extraction():
    client = instructor.from_claude_code(async_client=True)
    
    # This would require CLI streaming support
    async for partial_response in client.create_partial(
        response_model=MyModel,
        messages=messages
    ):
        print(f"Partial: {partial_response}")
```

## Migration Guide

### From Anthropic API to Claude Code CLI

```python
# Before (Anthropic API)
import anthropic
import instructor

anthropic_client = anthropic.Anthropic(api_key="your-key")
instructor_client = instructor.from_anthropic(anthropic_client)

# After (Claude Code CLI)
import instructor

instructor_client = instructor.from_claude_code()

# Usage remains the same!
response = instructor_client.create(
    response_model=MyModel,
    messages=messages
)
```

## Conclusion

Claude Code CLI integration provides a seamless way to use Claude's capabilities through Instructor without managing API keys directly. The integration maintains the same interface as other providers while leveraging the CLI's authentication and configuration management.

For production use, consider:
- Error handling and retry logic
- Monitoring and logging
- Performance implications of subprocess calls
- CLI availability and version compatibility

The integration is particularly useful for:
- Development environments
- Scripts and automation
- Situations where API key management is complex
- Leveraging existing Claude Code CLI configurations