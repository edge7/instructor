---
authors:
- jxnl
categories:
- Data Processing
- Data Transformation
comments: true
date: 2025-06-17
description: Transform messy, inconsistent JSON data into clean, validated data models using Instructor and Pydantic. Learn data cleaning patterns and validation techniques.
draft: false
slug: messy-json-clean-data-models
tags:
- Data Transformation
- JSON Processing
- Data Cleaning
- Pydantic
- Data Validation
- ETL
- Data Pipeline
- Instructor
---

# From Messy JSON to Clean Data Models

Real-world data is messy. APIs return inconsistent formats, user inputs contain typos, and legacy systems produce malformed JSON. Traditional data processing involves brittle parsing, manual cleaning, and endless edge case handling.

This comprehensive guide shows you how to transform chaotic data into clean, validated data models using LLMs and structured outputs. Learn battle-tested patterns for handling inconsistent formats, missing fields, and data quality issues.

<!-- more -->

## The Problem with Messy Data

### Common Data Quality Issues

Real-world JSON data comes with numerous challenges:

```python
# Inconsistent field names
data_source_1 = {"user_name": "John", "email_address": "john@example.com"}
data_source_2 = {"username": "Jane", "email": "jane@example.com"}
data_source_3 = {"name": "Bob", "mail": "bob@example.com"}

# Mixed data types
messy_data = {
    "age": "25",        # String instead of int
    "active": "true",   # String instead of bool
    "score": "null",    # String instead of null
    "tags": "tag1,tag2" # String instead of array
}

# Nested inconsistencies
complex_messy = {
    "user": {
        "personal_info": {
            "name": "John Doe",
            "age": "thirty-five"  # Written out instead of numeric
        }
    },
    "contact": "john@email.com, 555-123-4567"  # Mixed contact info
}

# Missing or null fields
incomplete_data = {
    "name": "Alice",
    "email": null,
    "phone": "",
    "address": {
        "street": null,
        "city": "Unknown"
    }
}
```

### Traditional Approaches Fall Short

```python
# Traditional manual parsing - brittle and error-prone
def parse_user_data(raw_data):
    try:
        name = raw_data.get("name") or raw_data.get("user_name") or raw_data.get("username")
        email = raw_data.get("email") or raw_data.get("email_address") or raw_data.get("mail")
        
        age_str = raw_data.get("age", "0")
        if age_str == "null" or age_str is None:
            age = None
        else:
            try:
                age = int(age_str)
            except (ValueError, TypeError):
                # Try to parse written numbers
                age_map = {"thirty-five": 35, "twenty-five": 25}
                age = age_map.get(age_str.lower(), 0)
        
        # ... hundreds more lines of parsing logic
        
    except Exception as e:
        print(f"Parsing failed: {e}")
        return None
```

This approach is:
- **Brittle**: Breaks with new data variations
- **Unmaintainable**: Complex parsing logic is hard to update
- **Error-prone**: Easy to miss edge cases
- **Unscalable**: Doesn't adapt to new data sources

## The LLM Solution: Intelligent Data Transformation

### Basic Data Cleaning with Structured Outputs

```python
import instructor
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from openai import OpenAI

class CleanUser(BaseModel):
    """Clean, validated user data model."""
    name: str = Field(..., min_length=1, description="Full name of the user")
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    age: Optional[int] = Field(None, ge=0, le=150)
    active: bool = Field(True, description="Whether the user account is active")
    tags: List[str] = Field(default_factory=list)
    
    @validator('name')
    def clean_name(cls, v):
        """Clean and normalize name."""
        return ' '.join(v.strip().title().split())
    
    @validator('email', pre=True)
    def extract_email(cls, v):
        """Extract email from mixed contact info."""
        if not v or v == "null":
            return None
        
        # If it's a string with mixed contact info
        if isinstance(v, str) and (',' in v or ' ' in v):
            import re
            email_match = re.search(r'[^@]+@[^@]+\.[^@]+', v)
            return email_match.group(0) if email_match else None
        
        return v
    
    @validator('tags', pre=True)
    def parse_tags(cls, v):
        """Parse tags from various formats."""
        if not v:
            return []
        
        if isinstance(v, str):
            # Handle comma-separated string
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        
        if isinstance(v, list):
            return [str(tag).strip() for tag in v if str(tag).strip()]
        
        return []

client = instructor.from_openai(OpenAI())

# Transform messy data into clean model
messy_json = """
{
    "user_name": "  john   doe  ",
    "email_address": "john@example.com, 555-123-4567",
    "age": "thirty-five",
    "is_active": "true",
    "user_tags": "developer,python,ai"
}
"""

clean_user = client.chat.completions.create(
    model="gpt-4",
    response_model=CleanUser,
    messages=[{
        "role": "user",
        "content": f"Clean and standardize this user data: {messy_json}"
    }]
)

print(f"Clean user: {clean_user}")
# Output: Clean user: CleanUser(name='John Doe', email='john@example.com', age=35, active=True, tags=['developer', 'python', 'ai'])
```

## Advanced Data Transformation Patterns

### 1. Multi-Source Data Harmonization

```python
from typing import Union, Dict, Any
from enum import Enum

class DataSource(str, Enum):
    API_V1 = "api_v1"
    API_V2 = "api_v2"
    LEGACY = "legacy"
    USER_INPUT = "user_input"

class ContactInfo(BaseModel):
    """Unified contact information model."""
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    
    @validator('phone', pre=True)
    def normalize_phone(cls, v):
        """Normalize phone numbers from various formats."""
        if not v or v == "null":
            return None
        
        # Extract digits only
        import re
        digits = re.sub(r'\D', '', str(v))
        
        if len(digits) == 10:
            return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
        
        return v

class UnifiedUser(BaseModel):
    """Unified user model that handles multiple data source formats."""
    id: str
    name: str
    contact: ContactInfo
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: DataSource
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @validator('name', pre=True)
    def extract_name(cls, v, values):
        """Extract name from various field formats."""
        if isinstance(v, dict):
            # Handle nested name objects
            first = v.get('first', v.get('firstName', ''))
            last = v.get('last', v.get('lastName', ''))
            return f"{first} {last}".strip()
        
        return str(v).strip()
    
    @validator('id', pre=True)
    def normalize_id(cls, v):
        """Normalize IDs from different systems."""
        return str(v).strip().lower()

class DataHarmonizer:
    """Harmonize data from multiple sources."""
    
    def __init__(self, client):
        self.client = client
    
    def harmonize_user_data(self, raw_data: Dict[str, Any], source: DataSource) -> UnifiedUser:
        """Convert messy multi-source data to unified format."""
        
        # Create source-specific context for the LLM
        source_instructions = {
            DataSource.API_V1: "This is from API v1 - fields are user_name, user_email, user_phone",
            DataSource.API_V2: "This is from API v2 - uses camelCase: userName, userEmail, userPhone",
            DataSource.LEGACY: "This is legacy data - inconsistent field names and formats",
            DataSource.USER_INPUT: "This is user input - may contain typos and inconsistent formatting"
        }
        
        instruction = source_instructions.get(source, "Clean and standardize this data")
        
        return self.client.chat.completions.create(
            model="gpt-4",
            response_model=UnifiedUser,
            messages=[{
                "role": "user",
                "content": f"""
                {instruction}
                
                Convert this data to a unified format:
                {raw_data}
                
                Source: {source.value}
                """
            }]
        )

# Example usage with different data sources
harmonizer = DataHarmonizer(client)

# API v1 format
api_v1_data = {
    "user_id": "USER123",
    "user_name": "john doe",
    "user_email": "JOHN@EXAMPLE.COM",
    "user_phone": "5551234567"
}

# API v2 format  
api_v2_data = {
    "userId": "user123",
    "userName": {"first": "John", "last": "Doe"},
    "userEmail": "john@example.com",
    "userPhone": "+1 (555) 123-4567"
}

# Legacy format
legacy_data = {
    "id": "123",
    "full_name": "John Doe",
    "contact_info": "john@example.com, 555-123-4567",
    "notes": "VIP customer"
}

# Harmonize all sources
unified_users = [
    harmonizer.harmonize_user_data(api_v1_data, DataSource.API_V1),
    harmonizer.harmonize_user_data(api_v2_data, DataSource.API_V2),
    harmonizer.harmonize_user_data(legacy_data, DataSource.LEGACY)
]

for user in unified_users:
    print(f"Unified: {user.name} ({user.source}) - Confidence: {user.confidence}")
```

### 2. Hierarchical Data Cleaning

```python
from typing import List, Optional, Dict

class Address(BaseModel):
    """Clean address model."""
    street: Optional[str] = None
    city: str
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "US"
    
    @validator('zip_code', pre=True)
    def clean_zip(cls, v):
        """Clean ZIP codes."""
        if not v or str(v).lower() in ['null', 'none', '']:
            return None
        
        # Extract digits and hyphens only
        import re
        cleaned = re.sub(r'[^\d-]', '', str(v))
        
        # Format as XXXXX or XXXXX-XXXX
        if len(cleaned) == 5:
            return cleaned
        elif len(cleaned) == 9:
            return f"{cleaned[:5]}-{cleaned[5:]}"
        elif len(cleaned) > 5:
            return cleaned[:5]  # Take first 5 digits
        
        return None
    
    @validator('state', pre=True)
    def normalize_state(cls, v):
        """Normalize state names and abbreviations."""
        if not v:
            return None
        
        state_map = {
            'california': 'CA', 'ca': 'CA', 'calif': 'CA',
            'new york': 'NY', 'ny': 'NY', 'newyork': 'NY',
            'texas': 'TX', 'tx': 'TX', 'tex': 'TX',
            # Add more mappings as needed
        }
        
        normalized = str(v).lower().strip()
        return state_map.get(normalized, v.upper() if len(v) == 2 else v.title())

class Company(BaseModel):
    """Clean company model."""
    name: str
    industry: Optional[str] = None
    size: Optional[str] = None
    website: Optional[str] = None
    
    @validator('name', pre=True)
    def clean_company_name(cls, v):
        """Clean company names."""
        if not v:
            raise ValueError("Company name is required")
        
        name = str(v).strip()
        
        # Common company suffixes
        suffixes = ['inc', 'corp', 'llc', 'ltd', 'co']
        
        for suffix in suffixes:
            # Remove suffix with various punctuation
            import re
            pattern = rf'\s*[,.]?\s*{suffix}\.?\s*$'
            name = re.sub(pattern, f' {suffix.upper()}', name, flags=re.IGNORECASE)
        
        return ' '.join(name.split())  # Normalize whitespace
    
    @validator('website', pre=True)
    def clean_website(cls, v):
        """Clean and validate website URLs."""
        if not v or str(v).lower() in ['null', 'none', '', 'n/a']:
            return None
        
        url = str(v).strip().lower()
        
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        # Basic URL validation
        import re
        if re.match(r'https?://[^\s/$.?#].[^\s]*$', url):
            return url
        
        return None

class CompleteProfile(BaseModel):
    """Complete user profile with nested data cleaning."""
    personal_info: UnifiedUser
    address: Optional[Address] = None
    company: Optional[Company] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('preferences', pre=True)
    def clean_preferences(cls, v):
        """Clean preference data."""
        if not v:
            return {}
        
        if isinstance(v, str):
            # Try to parse JSON string
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Parse key=value pairs
                prefs = {}
                for pair in v.split(','):
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        prefs[key.strip()] = value.strip()
                return prefs
        
        return v if isinstance(v, dict) else {}

# Complex nested data cleaning
complex_messy_data = """
{
    "user_info": {
        "id": "  U123  ",
        "full_name": "john   doe",
        "email": "JOHN@EXAMPLE.COM",
        "phone": "(555) 123-4567"
    },
    "address_info": {
        "street_address": "123 main st",
        "city_name": "san francisco",
        "state_code": "california",
        "postal_code": "94102-1234",
        "country_code": "usa"
    },
    "company_info": {
        "company_name": "tech corp, inc.",
        "industry_type": "software",
        "company_size": "50-100",
        "web_site": "www.techcorp.com"
    },
    "user_preferences": "theme=dark,notifications=true,language=en"
}
"""

clean_profile = client.chat.completions.create(
    model="gpt-4",
    response_model=CompleteProfile,
    messages=[{
        "role": "user",
        "content": f"Clean and structure this complex profile data: {complex_messy_data}"
    }]
)

print(f"Clean profile: {clean_profile}")
```

### 3. Batch Data Transformation

```python
import asyncio
from typing import List, Tuple
from instructor import AsyncInstructor
from openai import AsyncOpenAI

class BatchDataCleaner:
    """High-performance batch data cleaning."""
    
    def __init__(self, max_concurrent: int = 10):
        self.client = AsyncInstructor.from_openai(AsyncOpenAI())
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def clean_single_record(
        self, 
        record: Dict[str, Any], 
        target_model: BaseModel,
        record_id: Optional[str] = None
    ) -> Tuple[Optional[BaseModel], Optional[str]]:
        """Clean a single record with error handling."""
        
        async with self.semaphore:
            try:
                clean_record = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Faster for batch processing
                    response_model=target_model,
                    messages=[{
                        "role": "user",
                        "content": f"Clean and validate this data: {record}"
                    }],
                    max_retries=2
                )
                
                return clean_record, None
                
            except Exception as e:
                error_msg = f"Failed to clean record {record_id}: {str(e)}"
                return None, error_msg
    
    async def clean_batch(
        self, 
        records: List[Dict[str, Any]], 
        target_model: BaseModel
    ) -> Tuple[List[BaseModel], List[str]]:
        """Clean a batch of records concurrently."""
        
        tasks = [
            self.clean_single_record(record, target_model, f"record_{i}")
            for i, record in enumerate(records)
        ]
        
        results = await asyncio.gather(*tasks)
        
        successful_records = []
        errors = []
        
        for clean_record, error in results:
            if clean_record:
                successful_records.append(clean_record)
            if error:
                errors.append(error)
        
        return successful_records, errors
    
    def clean_csv_data(self, csv_file_path: str, target_model: BaseModel) -> Dict[str, Any]:
        """Clean data from CSV file."""
        import pandas as pd
        
        # Read CSV
        df = pd.read_csv(csv_file_path)
        
        # Convert to list of dicts
        raw_records = df.to_dict('records')
        
        # Clean batch
        clean_records, errors = asyncio.run(
            self.clean_batch(raw_records, target_model)
        )
        
        return {
            'clean_records': clean_records,
            'errors': errors,
            'success_rate': len(clean_records) / len(raw_records),
            'total_processed': len(raw_records)
        }

# Example batch processing
batch_cleaner = BatchDataCleaner(max_concurrent=5)

# Sample messy data
messy_records = [
    {"name": "john doe", "email": "JOHN@EXAMPLE.COM", "age": "25"},
    {"name": "jane smith", "email": "jane@test.com", "age": "thirty"},
    {"name": "bob wilson", "email": "invalid-email", "age": "null"},
    # ... hundreds more records
]

# Clean all records
clean_results = asyncio.run(
    batch_cleaner.clean_batch(messy_records, CleanUser)
)

clean_records, errors = clean_results

print(f"Successfully cleaned: {len(clean_records)} records")
print(f"Errors: {len(errors)} records")

for error in errors[:5]:  # Show first 5 errors
    print(f"Error: {error}")
```

## Data Quality Validation and Monitoring

### 1. Quality Metrics and Scoring

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics

@dataclass
class DataQualityScore:
    """Data quality assessment."""
    completeness: float  # 0-1, percentage of non-null fields
    accuracy: float      # 0-1, estimated accuracy of data
    consistency: float   # 0-1, consistency across records
    validity: float      # 0-1, adherence to validation rules
    overall: float       # 0-1, weighted average
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'validity': self.validity,
            'overall': self.overall
        }

class DataQualityAssessment(BaseModel):
    """LLM-based data quality assessment."""
    quality_score: DataQualityScore
    issues_found: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class QualityMonitor:
    """Monitor and assess data quality."""
    
    def __init__(self, client):
        self.client = client
    
    def assess_record_quality(
        self, 
        original_data: Dict[str, Any],
        cleaned_data: BaseModel
    ) -> DataQualityAssessment:
        """Assess quality of a single record transformation."""
        
        assessment_prompt = f"""
        Assess the data quality transformation:
        
        Original data: {original_data}
        Cleaned data: {cleaned_data.dict()}
        
        Evaluate:
        1. Completeness - how much original data was preserved
        2. Accuracy - how accurate the transformations appear
        3. Consistency - how consistent the format is
        4. Validity - how well it meets validation rules
        
        Identify any issues and provide recommendations.
        """
        
        return self.client.chat.completions.create(
            model="gpt-4",
            response_model=DataQualityAssessment,
            messages=[{"role": "user", "content": assessment_prompt}]
        )
    
    def assess_batch_quality(
        self, 
        original_batch: List[Dict[str, Any]],
        cleaned_batch: List[BaseModel]
    ) -> Dict[str, Any]:
        """Assess quality of batch transformation."""
        
        # Calculate aggregate metrics
        individual_assessments = []
        
        for original, cleaned in zip(original_batch, cleaned_batch):
            if cleaned:  # Only assess successful transformations
                assessment = self.assess_record_quality(original, cleaned)
                individual_assessments.append(assessment)
        
        if not individual_assessments:
            return {"error": "No successful transformations to assess"}
        
        # Aggregate scores
        completeness_scores = [a.quality_score.completeness for a in individual_assessments]
        accuracy_scores = [a.quality_score.accuracy for a in individual_assessments]
        consistency_scores = [a.quality_score.consistency for a in individual_assessments]
        validity_scores = [a.quality_score.validity for a in individual_assessments]
        overall_scores = [a.quality_score.overall for a in individual_assessments]
        
        # Collect all issues and recommendations
        all_issues = []
        all_recommendations = []
        
        for assessment in individual_assessments:
            all_issues.extend(assessment.issues_found)
            all_recommendations.extend(assessment.recommendations)
        
        return {
            'aggregate_scores': {
                'completeness': {
                    'mean': statistics.mean(completeness_scores),
                    'median': statistics.median(completeness_scores),
                    'min': min(completeness_scores),
                    'max': max(completeness_scores)
                },
                'accuracy': {
                    'mean': statistics.mean(accuracy_scores),
                    'median': statistics.median(accuracy_scores),
                    'min': min(accuracy_scores),
                    'max': max(accuracy_scores)
                },
                'overall': {
                    'mean': statistics.mean(overall_scores),
                    'median': statistics.median(overall_scores),
                    'min': min(overall_scores),
                    'max': max(overall_scores)
                }
            },
            'common_issues': list(set(all_issues)),
            'recommendations': list(set(all_recommendations)),
            'records_assessed': len(individual_assessments),
            'success_rate': len(cleaned_batch) / len(original_batch)
        }

# Usage
quality_monitor = QualityMonitor(client)

# Assess individual record
original_record = {"name": "john doe", "email": "JOHN@EXAMPLE.COM", "age": "25"}
cleaned_record = CleanUser(name="John Doe", email="john@example.com", age=25, active=True, tags=[])

quality_assessment = quality_monitor.assess_record_quality(original_record, cleaned_record)

print(f"Quality Score: {quality_assessment.quality_score.overall:.2f}")
print(f"Issues: {quality_assessment.issues_found}")
print(f"Recommendations: {quality_assessment.recommendations}")
```

### 2. Automated Data Validation Pipeline

```python
from typing import Callable, Any
from enum import Enum
import logging

class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationResult(BaseModel):
    """Result of a validation check."""
    field_name: str
    severity: ValidationSeverity
    message: str
    original_value: Any
    transformed_value: Any
    passed: bool

class DataValidationPipeline:
    """Comprehensive data validation pipeline."""
    
    def __init__(self, client):
        self.client = client
        self.validators: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def add_validator(self, validator_func: Callable):
        """Add a custom validator function."""
        self.validators.append(validator_func)
    
    def validate_email_format(self, original: str, transformed: str) -> ValidationResult:
        """Validate email format transformation."""
        import re
        
        email_pattern = r'^[^@]+@[^@]+\.[^@]+$'
        
        if not transformed:
            return ValidationResult(
                field_name="email",
                severity=ValidationSeverity.WARNING,
                message="Email was removed during transformation",
                original_value=original,
                transformed_value=transformed,
                passed=False
            )
        
        if not re.match(email_pattern, transformed):
            return ValidationResult(
                field_name="email",
                severity=ValidationSeverity.ERROR,
                message="Transformed email doesn't match valid format",
                original_value=original,
                transformed_value=transformed,
                passed=False
            )
        
        return ValidationResult(
            field_name="email",
            severity=ValidationSeverity.INFO,
            message="Email format validation passed",
            original_value=original,
            transformed_value=transformed,
            passed=True
        )
    
    def validate_data_preservation(
        self, 
        original: Dict[str, Any], 
        transformed: BaseModel
    ) -> List[ValidationResult]:
        """Validate that important data wasn't lost."""
        
        results = []
        transformed_dict = transformed.dict()
        
        # Check if critical fields were preserved
        critical_fields = ['name', 'email', 'id']
        
        for field in critical_fields:
            original_value = original.get(field)
            transformed_value = transformed_dict.get(field)
            
            if original_value and not transformed_value:
                results.append(ValidationResult(
                    field_name=field,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Critical field '{field}' was lost during transformation",
                    original_value=original_value,
                    transformed_value=transformed_value,
                    passed=False
                ))
            elif original_value and transformed_value:
                # Check if transformation preserved meaning
                similarity = self._check_semantic_similarity(original_value, transformed_value)
                
                if similarity < 0.8:  # 80% similarity threshold
                    results.append(ValidationResult(
                        field_name=field,
                        severity=ValidationSeverity.WARNING,
                        message=f"Field '{field}' transformation may have lost meaning",
                        original_value=original_value,
                        transformed_value=transformed_value,
                        passed=False
                    ))
        
        return results
    
    def _check_semantic_similarity(self, original: str, transformed: str) -> float:
        """Check semantic similarity between original and transformed values."""
        
        # Use LLM to assess semantic similarity
        similarity_check = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=SemanticSimilarity,
            messages=[{
                "role": "user",
                "content": f"""
                Rate the semantic similarity between these two values on a scale of 0.0 to 1.0:
                Original: "{original}"
                Transformed: "{transformed}"
                
                Consider:
                - Do they represent the same entity/concept?
                - Is the core meaning preserved?
                - Are they equivalent for business purposes?
                """
            }]
        )
        
        return similarity_check.similarity_score
    
    def run_pipeline(
        self, 
        original_data: Dict[str, Any],
        cleaned_data: BaseModel
    ) -> Dict[str, Any]:
        """Run complete validation pipeline."""
        
        all_results = []
        
        # Run data preservation validation
        preservation_results = self.validate_data_preservation(original_data, cleaned_data)
        all_results.extend(preservation_results)
        
        # Run custom validators
        for validator in self.validators:
            try:
                result = validator(original_data, cleaned_data)
                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)
            except Exception as e:
                self.logger.error(f"Validator failed: {e}")
        
        # Categorize results by severity
        results_by_severity = {
            ValidationSeverity.CRITICAL: [],
            ValidationSeverity.ERROR: [],
            ValidationSeverity.WARNING: [],
            ValidationSeverity.INFO: []
        }
        
        for result in all_results:
            results_by_severity[result.severity].append(result)
        
        # Calculate overall validation status
        has_critical = len(results_by_severity[ValidationSeverity.CRITICAL]) > 0
        has_errors = len(results_by_severity[ValidationSeverity.ERROR]) > 0
        
        if has_critical:
            overall_status = "FAILED_CRITICAL"
        elif has_errors:
            overall_status = "FAILED_ERROR"
        elif results_by_severity[ValidationSeverity.WARNING]:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "PASSED"
        
        return {
            'overall_status': overall_status,
            'results_by_severity': {k.value: v for k, v in results_by_severity.items()},
            'total_checks': len(all_results),
            'critical_count': len(results_by_severity[ValidationSeverity.CRITICAL]),
            'error_count': len(results_by_severity[ValidationSeverity.ERROR]),
            'warning_count': len(results_by_severity[ValidationSeverity.WARNING])
        }

class SemanticSimilarity(BaseModel):
    """Semantic similarity assessment."""
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)

# Usage
pipeline = DataValidationPipeline(client)

# Add custom validator
def validate_age_reasonableness(original: Dict[str, Any], cleaned: BaseModel) -> ValidationResult:
    """Validate that age transformations are reasonable."""
    
    original_age = original.get('age')
    cleaned_age = getattr(cleaned, 'age', None)
    
    if original_age and cleaned_age:
        # Check if age transformation is reasonable
        if isinstance(original_age, str) and original_age.lower() in ['thirty', 'thirty-five']:
            expected_range = (30, 40)
            if not (expected_range[0] <= cleaned_age <= expected_range[1]):
                return ValidationResult(
                    field_name="age",
                    severity=ValidationSeverity.ERROR,
                    message=f"Age transformation seems incorrect: {original_age} -> {cleaned_age}",
                    original_value=original_age,
                    transformed_value=cleaned_age,
                    passed=False
                )
    
    return ValidationResult(
        field_name="age",
        severity=ValidationSeverity.INFO,
        message="Age validation passed",
        original_value=original_age,
        transformed_value=cleaned_age,
        passed=True
    )

pipeline.add_validator(validate_age_reasonableness)

# Run validation
original_data = {"name": "john doe", "email": "JOHN@EXAMPLE.COM", "age": "thirty-five"}
cleaned_data = CleanUser(name="John Doe", email="john@example.com", age=35, active=True, tags=[])

validation_report = pipeline.run_pipeline(original_data, cleaned_data)

print(f"Validation Status: {validation_report['overall_status']}")
print(f"Critical Issues: {validation_report['critical_count']}")
print(f"Errors: {validation_report['error_count']}")
print(f"Warnings: {validation_report['warning_count']}")
```

## Production Data Processing Pipelines

### 1. Streaming Data Processing

```python
import asyncio
from typing import AsyncIterator
import json

class StreamingDataProcessor:
    """Process streaming data in real-time."""
    
    def __init__(self, client, target_model: BaseModel):
        self.client = client
        self.target_model = target_model
        self.processed_count = 0
        self.error_count = 0
    
    async def process_stream(
        self, 
        data_stream: AsyncIterator[str]
    ) -> AsyncIterator[BaseModel]:
        """Process streaming JSON data."""
        
        async for raw_data in data_stream:
            try:
                # Parse JSON
                data_dict = json.loads(raw_data)
                
                # Clean and validate
                cleaned = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_model=self.target_model,
                    messages=[{
                        "role": "user",
                        "content": f"Clean this data: {data_dict}"
                    }]
                )
                
                self.processed_count += 1
                yield cleaned
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to process record: {e}")
                continue
    
    async def process_with_batching(
        self, 
        data_stream: AsyncIterator[str],
        batch_size: int = 10
    ) -> AsyncIterator[List[BaseModel]]:
        """Process stream with batching for efficiency."""
        
        batch = []
        
        async for raw_data in data_stream:
            batch.append(raw_data)
            
            if len(batch) >= batch_size:
                # Process batch
                cleaned_batch = await self._process_batch(batch)
                yield cleaned_batch
                batch = []
        
        # Process remaining items
        if batch:
            cleaned_batch = await self._process_batch(batch)
            yield cleaned_batch
    
    async def _process_batch(self, batch: List[str]) -> List[BaseModel]:
        """Process a batch of records."""
        
        tasks = []
        
        for raw_data in batch:
            try:
                data_dict = json.loads(raw_data)
                task = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_model=self.target_model,
                    messages=[{
                        "role": "user",
                        "content": f"Clean this data: {data_dict}"
                    }]
                )
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"Failed to parse JSON: {e}")
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            successful_results = [
                result for result in results 
                if not isinstance(result, Exception)
            ]
            
            return successful_results
        
        return []

# Example streaming usage
async def simulate_data_stream():
    """Simulate a data stream."""
    sample_data = [
        '{"name": "john doe", "email": "JOHN@EXAMPLE.COM", "age": "25"}',
        '{"name": "jane smith", "email": "jane@test.com", "age": "thirty"}',
        '{"name": "bob wilson", "email": "bob@company.com", "age": "45"}',
        # ... more data
    ]
    
    for data in sample_data:
        yield data
        await asyncio.sleep(0.1)  # Simulate streaming delay

# Process the stream
processor = StreamingDataProcessor(
    AsyncInstructor.from_openai(AsyncOpenAI()),
    CleanUser
)

async def main():
    stream = simulate_data_stream()
    
    async for cleaned_record in processor.process_stream(stream):
        print(f"Processed: {cleaned_record}")
    
    print(f"Total processed: {processor.processed_count}")
    print(f"Total errors: {processor.error_count}")

# Run the streaming processor
# asyncio.run(main())
```

### 2. ETL Pipeline Integration

```python
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import pandas as pd

class ETLPipeline:
    """Complete ETL pipeline with LLM-powered data cleaning."""
    
    def __init__(self, client, config: Dict[str, Any]):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pipeline statistics
        self.stats = {
            'records_processed': 0,
            'records_cleaned': 0,
            'records_failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def extract_from_source(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from various sources."""
        
        source_type = source_config.get('type')
        
        if source_type == 'csv':
            df = pd.read_csv(source_config['path'])
            return df.to_dict('records')
        
        elif source_type == 'json':
            import json
            with open(source_config['path'], 'r') as f:
                return json.load(f)
        
        elif source_type == 'api':
            import requests
            response = requests.get(source_config['url'])
            return response.json()
        
        elif source_type == 'database':
            # Database extraction logic
            pass
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def transform_data(
        self, 
        raw_records: List[Dict[str, Any]], 
        target_model: BaseModel
    ) -> Tuple[List[BaseModel], List[Dict[str, Any]]]:
        """Transform data using LLM cleaning."""
        
        self.logger.info(f"Starting transformation of {len(raw_records)} records")
        
        cleaned_records = []
        failed_records = []
        
        for i, record in enumerate(raw_records):
            try:
                cleaned = self.client.chat.completions.create(
                    model=self.config.get('model', 'gpt-3.5-turbo'),
                    response_model=target_model,
                    messages=[{
                        "role": "user",
                        "content": f"Clean and validate this record: {record}"
                    }],
                    max_retries=self.config.get('max_retries', 2)
                )
                
                cleaned_records.append(cleaned)
                self.stats['records_cleaned'] += 1
                
                if i % 100 == 0:
                    self.logger.info(f"Processed {i} records")
                
            except Exception as e:
                self.logger.error(f"Failed to clean record {i}: {e}")
                failed_records.append({
                    'original_record': record,
                    'error': str(e),
                    'record_index': i
                })
                self.stats['records_failed'] += 1
            
            self.stats['records_processed'] += 1
        
        return cleaned_records, failed_records
    
    def load_to_destination(
        self, 
        cleaned_records: List[BaseModel], 
        destination_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load cleaned data to destination."""
        
        dest_type = destination_config.get('type')
        
        if dest_type == 'csv':
            # Convert to DataFrame and save
            df = pd.DataFrame([record.dict() for record in cleaned_records])
            df.to_csv(destination_config['path'], index=False)
            
            return {'status': 'success', 'records_loaded': len(cleaned_records)}
        
        elif dest_type == 'json':
            import json
            with open(destination_config['path'], 'w') as f:
                json.dump([record.dict() for record in cleaned_records], f, indent=2)
            
            return {'status': 'success', 'records_loaded': len(cleaned_records)}
        
        elif dest_type == 'database':
            # Database loading logic
            pass
        
        else:
            raise ValueError(f"Unsupported destination type: {dest_type}")
    
    def run_pipeline(
        self, 
        source_config: Dict[str, Any],
        target_model: BaseModel,
        destination_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run complete ETL pipeline."""
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # Extract
            self.logger.info("Starting data extraction")
            raw_records = self.extract_from_source(source_config)
            self.logger.info(f"Extracted {len(raw_records)} records")
            
            # Transform
            self.logger.info("Starting data transformation")
            cleaned_records, failed_records = self.transform_data(raw_records, target_model)
            self.logger.info(f"Cleaned {len(cleaned_records)} records, {len(failed_records)} failed")
            
            # Load
            self.logger.info("Starting data loading")
            load_result = self.load_to_destination(cleaned_records, destination_config)
            self.logger.info(f"Loaded {load_result.get('records_loaded', 0)} records")
            
            self.stats['end_time'] = datetime.now()
            
            return {
                'status': 'success',
                'statistics': self.stats,
                'failed_records': failed_records,
                'load_result': load_result,
                'duration': (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            }
            
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            self.logger.error(f"Pipeline failed: {e}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'statistics': self.stats,
                'duration': (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            }

# Example ETL pipeline usage
pipeline_config = {
    'model': 'gpt-4',
    'max_retries': 3,
    'batch_size': 50
}

etl_pipeline = ETLPipeline(client, pipeline_config)

# Source configuration
source_config = {
    'type': 'csv',
    'path': 'data/messy_users.csv'
}

# Destination configuration
destination_config = {
    'type': 'json',
    'path': 'data/clean_users.json'
}

# Run pipeline
result = etl_pipeline.run_pipeline(
    source_config=source_config,
    target_model=CleanUser,
    destination_config=destination_config
)

print(f"Pipeline Status: {result['status']}")
print(f"Duration: {result['duration']:.2f} seconds")
print(f"Records Processed: {result['statistics']['records_processed']}")
print(f"Success Rate: {result['statistics']['records_cleaned'] / result['statistics']['records_processed']:.2%}")
```

## Best Practices and Optimization

### 1. Performance Optimization

```python
# Use appropriate models for different tasks
TASK_MODEL_MAPPING = {
    'simple_cleaning': 'gpt-3.5-turbo',      # Fast, cheap
    'complex_validation': 'gpt-4',            # Accurate, expensive
    'batch_processing': 'gpt-3.5-turbo',     # Volume processing
    'quality_assessment': 'gpt-4'            # Detailed analysis
}

# Optimize prompts for better performance
def create_optimized_prompt(data: Dict[str, Any], target_model: BaseModel) -> str:
    """Create optimized prompts for data cleaning."""
    
    model_name = target_model.__name__
    field_descriptions = {
        field_name: field_info.description 
        for field_name, field_info in target_model.__fields__.items()
        if field_info.description
    }
    
    return f"""
    Clean and validate this data to match the {model_name} schema.
    
    Field requirements:
    {json.dumps(field_descriptions, indent=2)}
    
    Input data: {json.dumps(data)}
    
    Rules:
    - Preserve all meaningful information
    - Standardize formats consistently
    - Use null for missing required fields
    - Apply reasonable defaults where appropriate
    """

# Caching for repeated patterns
from functools import lru_cache
import hashlib

class CachedDataCleaner:
    """Data cleaner with intelligent caching."""
    
    def __init__(self, client):
        self.client = client
        self.cache = {}
    
    def _get_cache_key(self, data: Dict[str, Any], model_name: str) -> str:
        """Generate cache key for data pattern."""
        # Create hash of data structure (not values)
        structure = {k: type(v).__name__ for k, v in data.items()}
        structure_str = json.dumps(structure, sort_keys=True)
        return f"{model_name}_{hashlib.md5(structure_str.encode()).hexdigest()}"
    
    def clean_with_cache(
        self, 
        data: Dict[str, Any], 
        target_model: BaseModel
    ) -> BaseModel:
        """Clean data with caching for similar patterns."""
        
        cache_key = self._get_cache_key(data, target_model.__name__)
        
        # Check if we've seen this pattern before
        if cache_key in self.cache:
            # Use cached transformation logic
            return self._apply_cached_transformation(data, self.cache[cache_key])
        
        # New pattern - use LLM
        cleaned = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=target_model,
            messages=[{
                "role": "user",
                "content": create_optimized_prompt(data, target_model)
            }]
        )
        
        # Cache the transformation pattern
        self.cache[cache_key] = self._extract_transformation_pattern(data, cleaned)
        
        return cleaned
    
    def _extract_transformation_pattern(
        self, 
        original: Dict[str, Any], 
        cleaned: BaseModel
    ) -> Dict[str, Any]:
        """Extract transformation pattern for caching."""
        
        # This would contain transformation rules
        # Simplified for example
        return {
            'field_mappings': {},
            'transformations': {},
            'validation_rules': {}
        }
    
    def _apply_cached_transformation(
        self, 
        data: Dict[str, Any], 
        pattern: Dict[str, Any]
    ) -> BaseModel:
        """Apply cached transformation pattern."""
        
        # Apply cached transformation logic
        # Simplified for example
        pass
```

### 2. Error Recovery Strategies

```python
class RobustDataCleaner:
    """Data cleaner with advanced error recovery."""
    
    def __init__(self, client):
        self.client = client
        self.fallback_strategies = [
            self._try_partial_cleaning,
            self._try_field_by_field,
            self._try_minimal_cleaning
        ]
    
    def clean_with_recovery(
        self, 
        data: Dict[str, Any], 
        target_model: BaseModel
    ) -> Tuple[Optional[BaseModel], List[str]]:
        """Clean data with multiple fallback strategies."""
        
        errors = []
        
        # Try primary cleaning
        try:
            return self._primary_cleaning(data, target_model), errors
        except Exception as e:
            errors.append(f"Primary cleaning failed: {str(e)}")
        
        # Try fallback strategies
        for strategy in self.fallback_strategies:
            try:
                result = strategy(data, target_model)
                if result:
                    return result, errors
            except Exception as e:
                errors.append(f"Fallback strategy failed: {str(e)}")
        
        return None, errors
    
    def _primary_cleaning(self, data: Dict[str, Any], target_model: BaseModel) -> BaseModel:
        """Primary cleaning strategy."""
        return self.client.chat.completions.create(
            model="gpt-4",
            response_model=target_model,
            messages=[{
                "role": "user",
                "content": f"Clean this data: {data}"
            }]
        )
    
    def _try_partial_cleaning(self, data: Dict[str, Any], target_model: BaseModel) -> Optional[BaseModel]:
        """Try cleaning with partial model."""
        from instructor import Partial
        
        partial_result = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=Partial[target_model],
            messages=[{
                "role": "user",
                "content": f"Clean as much as possible: {data}"
            }]
        )
        
        # Convert partial to full model if enough fields are present
        if self._is_sufficiently_complete(partial_result, target_model):
            return partial_result
        
        return None
    
    def _try_field_by_field(self, data: Dict[str, Any], target_model: BaseModel) -> Optional[BaseModel]:
        """Try cleaning field by field."""
        
        cleaned_fields = {}
        
        for field_name, field_info in target_model.__fields__.items():
            if field_name in data:
                try:
                    # Clean individual field
                    cleaned_value = self._clean_individual_field(
                        data[field_name], 
                        field_info
                    )
                    cleaned_fields[field_name] = cleaned_value
                except Exception:
                    continue
        
        # Try to create model with cleaned fields
        try:
            return target_model(**cleaned_fields)
        except Exception:
            return None
    
    def _try_minimal_cleaning(self, data: Dict[str, Any], target_model: BaseModel) -> Optional[BaseModel]:
        """Try minimal cleaning with default values."""
        
        # Create model with minimal required fields
        required_fields = {
            name: field_info 
            for name, field_info in target_model.__fields__.items()
            if field_info.required
        }
        
        minimal_data = {}
        
        for field_name, field_info in required_fields.items():
            if field_name in data and data[field_name]:
                minimal_data[field_name] = str(data[field_name]).strip()
            else:
                # Use default or placeholder
                if field_info.type_ == str:
                    minimal_data[field_name] = "Unknown"
                elif field_info.type_ == int:
                    minimal_data[field_name] = 0
                elif field_info.type_ == float:
                    minimal_data[field_name] = 0.0
                elif field_info.type_ == bool:
                    minimal_data[field_name] = False
        
        try:
            return target_model(**minimal_data)
        except Exception:
            return None
```

## Conclusion

Transforming messy JSON to clean data models with LLMs revolutionizes data processing:

### Key Benefits:
- **Intelligent Parsing**: LLMs understand context and intent
- **Adaptive Cleaning**: Handles variations without explicit rules
- **Semantic Validation**: Preserves meaning during transformation
- **Error Recovery**: Multiple fallback strategies for robustness

### Best Practices:
1. **Design Clear Models**: Use descriptive fields and validation rules
2. **Implement Quality Monitoring**: Track transformation accuracy
3. **Use Appropriate Models**: Balance cost and accuracy
4. **Cache Common Patterns**: Optimize for performance
5. **Plan Error Recovery**: Handle edge cases gracefully

### Production Considerations:
- Monitor data quality metrics continuously
- Implement comprehensive validation pipelines
- Use streaming processing for large datasets
- Cache transformation patterns for efficiency
- Plan for graceful degradation

This approach transforms data processing from brittle rule-based systems to intelligent, adaptive pipelines that grow with your data complexity.

## Related Concepts

- [Structured Output from LLMs: The Complete Guide](structured-output-llm-complete-guide.md) - Foundation concepts
- [Build Type-Safe AI Apps with Instructor + Pydantic](type-safe-ai-apps-instructor-pydantic.md) - Advanced validation patterns
- [Data Validation with Pydantic](../../concepts/validation.md) - Comprehensive validation strategies

## See Also

- [10 Instructor Patterns That Save Hours](instructor-patterns-save-hours.md) - Advanced techniques
- [ETL Examples](../../examples/extracting_tables.md) - Real-world data extraction
- [Validation Examples](../../examples/classification.md) - Classification and validation patterns

Start transforming your messy data today with [Instructor](https://github.com/jxnl/instructor)!