"""
Example demonstrating how to use the Together client directly with instructor.

This example shows how to use the Together Python package client directly
instead of manually creating an OpenAI client with Together's base URL.
"""

import os
from pydantic import BaseModel
import instructor

# Option 1: Using Together client directly (new way)
try:
    from together import Together
    
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    instructor_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    
    class UserExtract(BaseModel):
        name: str
        age: int

    user: UserExtract = instructor_client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )

    print("Using Together client directly:")
    print(user.model_dump_json(indent=2))
    
except ImportError:
    print("Together package not installed. Install with: pip install together")

# Option 2: Traditional way using OpenAI client (still works)
import openai

client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ.get("TOGETHER_API_KEY"),
)

instructor_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

class UserExtract(BaseModel):
    name: str
    age: int

user: UserExtract = instructor_client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract sarah is 30 years old"},
    ],
)

print("\nUsing OpenAI client with Together base URL:")
print(user.model_dump_json(indent=2))