"""
Example of using Instructor with xAI SDK for structured outputs
"""

import os
from typing import List
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI


# Example 1: JSON Mode - Extracting structured data
class UserProfile(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")
    interests: List[str] = Field(description="List of user interests")


# Example 2: Function Calling - Complex extraction
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str = Field(description="Postal or ZIP code")


class Company(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry or sector")
    founded_year: int = Field(description="Year the company was founded")
    headquarters: Address = Field(description="Company headquarters location")
    employee_count: int = Field(description="Approximate number of employees")


class ContactInfo(BaseModel):
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")


def example_json_mode():
    """Example using JSON mode for simple extraction"""
    print("=== xAI JSON Mode Example ===\n")
    
    # Create xAI client using OpenAI SDK
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    
    # Patch with instructor for JSON mode
    instructor_client = instructor.from_xai(
        client, 
        mode=instructor.Mode.XAI_JSON
    )
    
    # Extract structured data
    response = instructor_client.chat.completions.create(
        model="grok-beta",
        messages=[
            {
                "role": "user",
                "content": "Extract user info: John Doe is 28 years old and likes hiking, photography, and cooking."
            }
        ],
        response_model=UserProfile,
    )
    
    print(f"Extracted User Profile:")
    print(f"  Name: {response.name}")
    print(f"  Age: {response.age}")
    print(f"  Interests: {', '.join(response.interests)}")


def example_function_calling():
    """Example using function calling mode for complex extraction"""
    print("\n=== xAI Function Calling Mode Example ===\n")
    
    # Create xAI client
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    
    # Patch with instructor for function calling mode (default)
    instructor_client = instructor.from_xai(client)
    
    # Extract complex structured data
    response = instructor_client.chat.completions.create(
        model="grok-beta",
        messages=[
            {
                "role": "user",
                "content": """
                Extract company information from this text:
                
                Tesla, Inc. is an American electric vehicle and clean energy company 
                founded in 2003. The company is headquartered at 1 Tesla Road, Austin, 
                Texas 78725, USA. Tesla operates in the automotive and energy storage 
                industry and employs approximately 127,000 people worldwide.
                """
            }
        ],
        response_model=Company,
    )
    
    print(f"Extracted Company Information:")
    print(f"  Name: {response.name}")
    print(f"  Industry: {response.industry}")
    print(f"  Founded: {response.founded_year}")
    print(f"  Employees: {response.employee_count:,}")
    print(f"  Headquarters:")
    print(f"    Street: {response.headquarters.street}")
    print(f"    City: {response.headquarters.city}")
    print(f"    Country: {response.headquarters.country}")
    print(f"    Postal Code: {response.headquarters.postal_code}")


def example_using_auto_client():
    """Example using the auto client feature"""
    print("\n=== xAI Auto Client Example ===\n")
    
    # Using the auto client
    instructor_client = instructor.from_provider("xai/grok-beta")
    
    # Simple extraction
    response = instructor_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "My email is john@example.com and my phone is +1-555-0123"
            }
        ],
        response_model=ContactInfo,
    )
    
    print(f"Extracted Contact Info:")
    print(f"  Email: {response.email}")
    print(f"  Phone: {response.phone}")


if __name__ == "__main__":
    # Make sure to set your XAI_API_KEY environment variable
    if not os.environ.get("XAI_API_KEY"):
        print("Please set your XAI_API_KEY environment variable")
        print("export XAI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        example_json_mode()
        example_function_calling()
        example_using_auto_client()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set your XAI_API_KEY environment variable")
        print("2. Installed required packages: pip install openai instructor")