#!/usr/bin/env python3
"""
Example showing how to use Google GenAI SDK with Instructor for multimodal inputs.

This example demonstrates:
1. How to use instructor.from_genai() directly
2. How to use instructor.from_provider() 
3. Processing audio files (mp4 format and others)
4. Processing images and PDFs
5. Different loading methods (URL, path, base64)

To run this example:
1. Install dependencies: pip install "instructor[google-genai]"
2. Set your API key: export GOOGLE_API_KEY="your-api-key"
3. Run: python examples/genai_multimodal_example.py
"""

import instructor
from instructor.multimodal import Audio, Image, PDF
from pydantic import BaseModel, Field
from google.genai import Client
import os


# Define response models
class AudioAnalysis(BaseModel):
    """Analysis of an audio file"""
    transcript: str = Field(description="Full transcript of the audio")
    summary: str = Field(description="Brief summary of the content")
    key_points: list[str] = Field(description="Main points discussed")
    speakers: list[str] = Field(description="Identified speakers if any")
    duration_estimate: str = Field(description="Estimated duration")


class ImageDescription(BaseModel):
    """Description of an image"""
    objects: list[str] = Field(description="Objects visible in the image")
    scene: str = Field(description="Description of the scene")
    colors: list[str] = Field(description="Prominent colors")
    text_content: str = Field(description="Any text visible in the image")


class DocumentSummary(BaseModel):
    """Summary of a document"""
    title: str = Field(description="Document title")
    summary: str = Field(description="Brief summary")
    key_sections: list[str] = Field(description="Main sections")
    important_numbers: list[str] = Field(description="Important numbers or amounts")


def method_1_direct_from_genai():
    """Method 1: Using instructor.from_genai() directly"""
    print("=== Method 1: Direct from_genai usage ===")
    
    # Create the genai client
    genai_client = Client()
    
    # Patch with instructor
    client = instructor.from_genai(genai_client)
    
    # Process audio file
    audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"
    
    response = client.chat.completions.create(
        model="models/gemini-1.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    "Please analyze this audio file:",
                    Audio.from_url(audio_url),
                ],
            },
        ],
        response_model=AudioAnalysis,
    )
    
    print(f"Transcript: {response.transcript[:100]}...")
    print(f"Summary: {response.summary}")
    print(f"Key points: {response.key_points}")
    print()


def method_2_from_provider():
    """Method 2: Using instructor.from_provider() (recommended)"""
    print("=== Method 2: from_provider usage (recommended) ===")
    
    # Use from_provider - this is the recommended approach
    client = instructor.from_provider("genai/gemini-1.5-flash")
    
    # Process audio file
    audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    "Please analyze this audio file:",
                    Audio.from_url(audio_url),
                ],
            },
        ],
        response_model=AudioAnalysis,
    )
    
    print(f"Transcript: {response.transcript[:100]}...")
    print(f"Summary: {response.summary}")
    print(f"Key points: {response.key_points}")
    print()


def multimodal_examples():
    """Examples of different multimodal inputs"""
    print("=== Multimodal Examples ===")
    
    client = instructor.from_provider("genai/gemini-1.5-flash")
    
    # Audio from URL
    print("1. Audio from URL:")
    audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    "Transcribe this audio:",
                    Audio.from_url(audio_url),
                ],
            },
        ],
        response_model=AudioAnalysis,
    )
    print(f"   Audio summary: {response.summary}")
    
    # Audio from local file (if you have one)
    print("\n2. Audio from local file:")
    print("   # Audio.from_path('/path/to/your/audio.mp4')  # Supports mp4, wav, mp3, etc.")
    
    # Image processing
    print("\n3. Image processing:")
    image_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    "Describe this image:",
                    Image.from_url(image_url),
                ],
            },
        ],
        response_model=ImageDescription,
    )
    print(f"   Image description: {response.scene}")
    print(f"   Objects: {response.objects}")
    
    # PDF processing
    print("\n4. PDF processing:")
    pdf_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    "Analyze this PDF:",
                    PDF.from_url(pdf_url),
                ],
            },
        ],
        response_model=DocumentSummary,
    )
    print(f"   Document summary: {response.summary}")
    print(f"   Important numbers: {response.important_numbers}")
    print()


def audio_format_examples():
    """Examples with different audio formats"""
    print("=== Audio Format Examples ===")
    
    client = instructor.from_provider("genai/gemini-1.5-flash")
    
    print("Supported audio formats:")
    print("  - MP4 audio (audio/mp4)")
    print("  - WAV (audio/wav)")
    print("  - MP3 (audio/mp3)")
    print("  - AAC (audio/aac)")
    print("  - FLAC (audio/flac)")
    print("  - M4A (audio/m4a)")
    print("  - And more...")
    
    # Example of how to load different formats
    print("\nLoading examples:")
    print("  Audio.from_path('/path/to/audio.mp4')  # MP4 audio")
    print("  Audio.from_path('/path/to/audio.wav')  # WAV audio")
    print("  Audio.from_path('/path/to/audio.mp3')  # MP3 audio")
    print("  Audio.from_url('https://example.com/audio.m4a')  # M4A from URL")
    print()


def async_example():
    """Example of async usage"""
    print("=== Async Example ===")
    print("For async usage:")
    print("""
import asyncio
import instructor
from instructor.multimodal import Audio
from pydantic import BaseModel

async def analyze_audio():
    client = instructor.from_provider(
        "genai/gemini-1.5-flash", 
        async_client=True
    )
    
    response = await client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": [
                "Analyze this audio:",
                Audio.from_url("https://example.com/audio.mp4"),
            ],
        }],
        response_model=AudioAnalysis,
    )
    return response

# Run with: asyncio.run(analyze_audio())
""")
    print()


def main():
    """Main function demonstrating all examples"""
    print("🎵 Google GenAI SDK Multimodal Example with Instructor")
    print("=" * 60)
    
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("   export GOOGLE_API_KEY='your-api-key-here'")
        print("\nOr set it in your Python code:")
        print("   import os")
        print("   os.environ['GOOGLE_API_KEY'] = 'your-api-key-here'")
        return
    
    try:
        # Show both methods
        method_1_direct_from_genai()
        method_2_from_provider()
        
        # Show multimodal examples
        multimodal_examples()
        
        # Show audio format support
        audio_format_examples()
        
        # Show async example
        async_example()
        
        print("✅ All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. Use instructor.from_provider('genai/gemini-1.5-flash') for simplest setup")
        print("2. Use instructor.from_genai(client) for more control")
        print("3. Audio.from_path() supports MP4 and many other formats")
        print("4. Audio.from_url() works with remote files")
        print("5. Both sync and async are supported")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the required dependencies:")
        print("   pip install 'instructor[google-genai]'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your GOOGLE_API_KEY is valid and you have the required permissions.")


if __name__ == "__main__":
    main()