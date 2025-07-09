# Migration Guide: from_gemini to from_genai

This guide shows how to migrate from the deprecated `from_gemini` to the new `from_genai` approach for multimodal processing.

## Key Changes

### Old Way (Deprecated)
```python
import instructor
import google.generativeai as genai

# Old approach with google.generativeai
client = instructor.from_gemini(
    genai.GenerativeModel("models/gemini-1.5-flash")
)

# Upload file using old API
file = genai.upload_file("audio.mp4")
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": ["Analyze this audio:", file],
    }],
    response_model=YourModel,
)
```

### New Way (Recommended)
```python
import instructor
from instructor.multimodal import Audio, Image, PDF
from google.genai import Client

# Method 1: Direct from_genai usage
genai_client = Client()
client = instructor.from_genai(genai_client)

# Method 2: from_provider (simplest)
client = instructor.from_provider("genai/gemini-1.5-flash")

# Use instructor's multimodal classes
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            "Analyze this audio:",
            Audio.from_path("audio.mp4"),  # or Audio.from_url()
        ],
    }],
    response_model=YourModel,
)
```

## Multimodal Migration Examples

### Audio Processing

#### Old Way
```python
import google.generativeai as genai
import instructor

client = instructor.from_gemini(
    genai.GenerativeModel("models/gemini-1.5-flash")
)

# Upload file first
file = genai.upload_file("audio.mp4")
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": ["Transcribe this:", file],
    }],
    response_model=AudioTranscript,
)
```

#### New Way
```python
import instructor
from instructor.multimodal import Audio

# Much simpler!
client = instructor.from_provider("genai/gemini-1.5-flash")

response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            "Transcribe this:",
            Audio.from_path("audio.mp4"),  # Direct file loading
        ],
    }],
    response_model=AudioTranscript,
)
```

### Image Processing

#### Old Way
```python
import google.generativeai as genai
import instructor

client = instructor.from_gemini(
    genai.GenerativeModel("models/gemini-1.5-flash")
)

# Upload image
image = genai.upload_file("image.jpg")
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": ["Describe this image:", image],
    }],
    response_model=ImageDescription,
)
```

#### New Way
```python
import instructor
from instructor.multimodal import Image

client = instructor.from_provider("genai/gemini-1.5-flash")

response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            "Describe this image:",
            Image.from_path("image.jpg"),  # or Image.from_url()
        ],
    }],
    response_model=ImageDescription,
)
```

## Benefits of the New Approach

1. **Unified API**: Works consistently across all providers (OpenAI, Anthropic, Google, etc.)
2. **No File Upload Required**: Direct file loading from paths or URLs
3. **Better Type Safety**: Proper typing with the new google.genai SDK
4. **More Flexible**: Support for URLs, local files, and base64 data
5. **Future-Proof**: Built on Google's recommended SDK

## Supported Audio Formats

The new approach supports many audio formats:
- MP4 audio (`audio/mp4`)
- WAV (`audio/wav`)
- MP3 (`audio/mp3`)
- AAC (`audio/aac`)
- FLAC (`audio/flac`)
- M4A (`audio/m4a`)
- And more...

## Complete Example

Here's a complete working example for audio analysis:

```python
import instructor
from instructor.multimodal import Audio
from pydantic import BaseModel, Field
from google.genai import Client
import os

# Set your API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

class AudioAnalysis(BaseModel):
    transcript: str = Field(description="Full transcript")
    summary: str = Field(description="Brief summary")
    key_points: list[str] = Field(description="Main points")

# Method 1: Direct from_genai
genai_client = Client()
client = instructor.from_genai(genai_client)

# Method 2: from_provider (recommended)
client = instructor.from_provider("genai/gemini-1.5-flash")

# Analyze audio file
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            "Please analyze this audio file:",
            Audio.from_path("your_audio.mp4"),
            # or Audio.from_url("https://example.com/audio.mp4")
        ],
    }],
    response_model=AudioAnalysis,
)

print(f"Transcript: {response.transcript}")
print(f"Summary: {response.summary}")
print(f"Key points: {response.key_points}")
```

## Installation

Make sure you have the right dependencies:

```bash
# Install with genai support
pip install "instructor[google-genai]"

# If you were using the old approach, you can remove:
# pip uninstall google-generativeai
```

## Troubleshooting

1. **Import Error**: Make sure you have `google-genai` installed, not `google-generativeai`
2. **API Key**: Use `GOOGLE_API_KEY` environment variable
3. **File Formats**: Use the `Audio`, `Image`, and `PDF` classes from `instructor.multimodal`
4. **Async**: Use `async_client=True` with `from_provider()` for async support

The new approach is much simpler and more consistent. Give it a try!