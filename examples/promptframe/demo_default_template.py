#!/usr/bin/env python3
"""
Demo script showing the new default XML template functionality in promptframe.

This demonstrates how you can now call map_prompt without providing a template,
and it will automatically generate an XML template that wraps each column.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from pydantic import BaseModel, Field


class PersonInfo(BaseModel):
    """Schema for extracting person information."""
    full_name: str = Field(description="Full name of the person")
    age_group: str = Field(description="Age group: child, teen, adult, senior")
    profession: str = Field(description="Person's job or profession")
    location: str = Field(description="City or location")


class ProductInfo(BaseModel):
    """Schema for extracting product information."""
    category: str = Field(description="Product category")
    price_range: str = Field(description="Price range: budget, mid-range, premium")
    target_audience: str = Field(description="Who this product is for")


def demo_default_xml_template():
    """Demonstrate the default XML template functionality."""
    print("🚀 PromptFrame Demo - Default XML Template")
    print("=" * 60)
    
    # Import after adding to path
    from promptframe import PromptFrame
    from promptframe.utils import generate_default_xml_template
    
    # Sample person data
    person_df = pd.DataFrame({
        "name": ["Alice Johnson", "Bob Smith", "Carol Davis"],
        "age": [28, 45, 34],
        "job": ["Software Engineer", "Marketing Manager", "Data Scientist"],
        "city": ["San Francisco", "New York", "Seattle"]
    })
    
    print("Sample Person Data:")
    print(person_df)
    print()
    
    # Show what the default template looks like
    default_template = generate_default_xml_template(person_df)
    print("Generated Default XML Template:")
    print("-" * 40)
    print(default_template)
    print("-" * 40)
    print()
    
    # Create PromptFrame and use default template (no template parameter!)
    print("Processing with DEFAULT XML template (no template specified)...")
    pf = PromptFrame(person_df)
    
    # This will use the default XML template automatically!
    pf.map_prompt(
        name="person_info",
        schema=PersonInfo,
        # Note: No template parameter! It will auto-generate XML template
        progress=True
    )
    
    result = pf.to_df()
    print("\nResults with default XML template:")
    print(result)
    print()
    
    # Show the extracted information
    print("Extracted Information:")
    for i, row in result.iterrows():
        print(f"Row {i + 1}: {row['name']}")
        print(f"  Full Name: {row['person_info.full_name']}")
        print(f"  Age Group: {row['person_info.age_group']}")
        print(f"  Profession: {row['person_info.profession']}")
        print(f"  Location: {row['person_info.location']}")
        print()


def demo_custom_vs_default():
    """Compare custom template vs default template."""
    print("🔄 Comparison: Custom Template vs Default Template")
    print("=" * 60)
    
    from promptframe import PromptFrame
    
    # Product data with different column types
    product_df = pd.DataFrame({
        "product_name": ["iPhone 15", "Gaming Laptop", "Coffee Maker"],
        "price": [999, 1299, 89],
        "description": [
            "Latest smartphone with advanced camera",
            "High-performance laptop for gaming",
            "Automatic drip coffee maker with timer"
        ],
        "brand": ["Apple", "ASUS", "Cuisinart"]
    })
    
    print("Sample Product Data:")
    print(product_df)
    print()
    
    # Method 1: Custom template
    print("Method 1: Using CUSTOM template")
    pf1 = PromptFrame(product_df)
    pf1.map_prompt(
        name="custom_analysis",
        template="""
        Product: {{ product_name }}
        Price: ${{ price }}
        Description: {{ description }}
        Brand: {{ brand }}
        
        Analyze this product information and extract key details.
        """,
        schema=ProductInfo,
        progress=True
    )
    
    # Method 2: Default XML template
    print("\nMethod 2: Using DEFAULT XML template")
    pf2 = PromptFrame(product_df)
    pf2.map_prompt(
        name="default_analysis",
        schema=ProductInfo,
        # No template specified - uses default XML template!
        progress=True
    )
    
    print("\nResults comparison:")
    result1 = pf1.to_df()
    result2 = pf2.to_df()
    
    print("\nCustom template results:")
    for col in result1.columns:
        if col.startswith("custom_analysis."):
            print(f"  {col}: {result1[col].tolist()}")
    
    print("\nDefault template results:")
    for col in result2.columns:
        if col.startswith("default_analysis."):
            print(f"  {col}: {result2[col].tolist()}")


def demo_column_sanitization():
    """Demonstrate how column names with special characters are handled."""
    print("🧹 Demo: Column Name Sanitization")
    print("=" * 60)
    
    from promptframe import PromptFrame
    from promptframe.utils import generate_default_xml_template
    
    # DataFrame with challenging column names
    messy_df = pd.DataFrame({
        "user name": ["Alice", "Bob"],
        "email-address": ["alice@test.com", "bob@test.com"], 
        "signup.date": ["2023-01-15", "2023-02-20"],
        "is-active": [True, False]
    })
    
    print("DataFrame with challenging column names:")
    print(messy_df)
    print()
    
    # Show how the default template handles special characters
    template = generate_default_xml_template(messy_df)
    print("Generated XML template (note the sanitized tag names):")
    print("-" * 50)
    print(template)
    print("-" * 50)
    print()
    
    print("✅ Column names are automatically sanitized for XML:")
    print("  'user name' → <user_name>")
    print("  'email-address' → <email_address>")
    print("  'signup.date' → <signup_date>")
    print("  'is-active' → <is_active>")


def main():
    """Run all demos."""
    print("🎯 PromptFrame Default XML Template Demos")
    print("=" * 70)
    print("These demos show the new default XML template functionality!")
    print("Now you can call map_prompt() without specifying a template.\n")
    
    try:
        demo_default_xml_template()
        print("\n" + "="*70 + "\n")
        
        demo_custom_vs_default()
        print("\n" + "="*70 + "\n")
        
        demo_column_sanitization()
        
        print("\n" + "="*70)
        print("✅ All default template demos completed!")
        print("\n💡 Key Benefits:")
        print("   • No need to write templates for simple extraction")
        print("   • Automatic XML wrapping of all columns")
        print("   • Column name sanitization for valid XML")
        print("   • Still supports custom templates when needed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Installed required dependencies")
        raise


if __name__ == "__main__":
    main()