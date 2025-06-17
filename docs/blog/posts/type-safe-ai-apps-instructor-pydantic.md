---
authors:
- jxnl
categories:
- Type Safety
- Best Practices
comments: true
date: 2025-06-17
description: Learn to build robust, type-safe AI applications with Instructor and Pydantic. Master validation, error handling, and production-ready patterns.
draft: false
slug: type-safe-ai-apps-instructor-pydantic
tags:
- Type Safety
- Pydantic
- Instructor
- Validation
- Python
- AI Applications
- Error Handling
- Production
---

# Build Type-Safe AI Apps with Instructor + Pydantic

Type safety transforms unreliable LLM outputs into robust, production-ready applications. This comprehensive guide shows you how to leverage Instructor and Pydantic to build AI applications that fail fast, validate early, and maintain data integrity throughout your pipeline.

Learn the patterns, practices, and techniques that separate prototype code from production systems.

<!-- more -->

## Why Type Safety Matters in AI Applications

Traditional LLM integrations are fragile. String parsing, manual JSON validation, and hope-driven development lead to systems that break in production. Type safety provides:

### 1. Compile-Time Error Detection
```python
# Without type safety - runtime failures
def process_user_data(data: dict):
    name = data.get("name")  # Could be None
    age = data.get("age")    # Could be string, int, or None
    if age > 18:             # TypeError if age is None or string
        return f"Adult user: {name}"

# With type safety - guaranteed structure
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

def process_user_data(user: User):
    if user.age > 18:  # Always safe - age is guaranteed to be int
        return f"Adult user: {user.name}"
```

### 2. IDE Support and Developer Experience
```python
# Full autocomplete, refactoring support, and error detection
user = client.chat.completions.create(
    model="gpt-4",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 25"}]
)

# IDE knows exactly what properties are available
print(user.name)        # ✅ IDE autocomplete
print(user.age)         # ✅ IDE autocomplete
print(user.invalid)     # ❌ IDE error before runtime
```

### 3. Runtime Validation
```python
class EmailAddress(BaseModel):
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    verified: bool = False

# Automatic validation on creation
try:
    email = EmailAddress(email="invalid-email", verified=True)
except ValidationError as e:
    print("Invalid email format")  # Caught immediately
```

## Foundation: Building Your First Type-Safe Model

### Basic Model Structure

```python
import instructor
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from openai import OpenAI

class Contact(BaseModel):
    """A validated contact extracted from text."""
    name: str = Field(..., min_length=1, description="Full name of the contact")
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    phone: Optional[str] = Field(None, regex=r'^\+?1?\d{9,15}$')
    company: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    
client = instructor.from_openai(OpenAI())

contact = client.chat.completions.create(
    model="gpt-4",
    response_model=Contact,
    messages=[{
        "role": "user", 
        "content": "Extract contact: John Doe, john@example.com, works at Acme Corp"
    }]
)

# Guaranteed valid Contact object
print(f"Valid contact: {contact.name} ({contact.email})")
```

### Advanced Validation Patterns

```python
from pydantic import validator, root_validator
from typing import Union
import re

class BusinessCard(BaseModel):
    """Advanced validation for business card extraction."""
    name: str = Field(..., min_length=2)
    title: Optional[str] = None
    company: str = Field(..., min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower() if v else None
    
    @validator('phone')
    def normalize_phone(cls, v):
        if v:
            # Remove all non-digits
            digits = re.sub(r'\D', '', v)
            if len(digits) < 10:
                raise ValueError('Phone number too short')
            return f"+1-{digits[-10:-7]}-{digits[-7:-4]}-{digits[-4:]}"
        return v
    
    @validator('website')
    def validate_website(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            return f"https://{v}"
        return v
    
    @root_validator
    def validate_contact_info(cls, values):
        email, phone = values.get('email'), values.get('phone')
        if not email and not phone:
            raise ValueError('Must have either email or phone')
        return values

# Usage with automatic validation
card = client.chat.completions.create(
    model="gpt-4",
    response_model=BusinessCard,
    messages=[{
        "role": "user",
        "content": "Extract from business card: Jane Smith, CEO, Tech Solutions Inc, jane@techsolutions.com, 555-123-4567"
    }]
)

print(f"Validated card: {card.name} at {card.company}")
print(f"Contact: {card.email} or {card.phone}")
```

## Production Patterns

### 1. Hierarchical Data Models

```python
from typing import List, Dict, Any
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Task(BaseModel):
    """Individual task with validation."""
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: Priority = Priority.MEDIUM
    estimated_hours: Optional[float] = Field(None, ge=0.1, le=100.0)
    tags: List[str] = Field(default_factory=list)
    
    @validator('tags')
    def validate_tags(cls, v):
        return [tag.lower().strip() for tag in v if tag.strip()]

class ProjectPlan(BaseModel):
    """Complete project plan with nested validation."""
    project_name: str = Field(..., min_length=3)
    description: str
    tasks: List[Task] = Field(..., min_items=1)
    total_estimated_hours: Optional[float] = None
    
    @root_validator
    def calculate_total_hours(cls, values):
        tasks = values.get('tasks', [])
        total = sum(task.estimated_hours or 0 for task in tasks)
        values['total_estimated_hours'] = total if total > 0 else None
        return values
    
    def get_high_priority_tasks(self) -> List[Task]:
        return [task for task in self.tasks if task.priority in [Priority.HIGH, Priority.URGENT]]
    
    def get_tasks_by_tag(self, tag: str) -> List[Task]:
        return [task for task in self.tasks if tag.lower() in task.tags]

# Extract complex nested data
project = client.chat.completions.create(
    model="gpt-4",
    response_model=ProjectPlan,
    messages=[{
        "role": "user",
        "content": """
        Create a project plan for building a mobile app:
        - Design user interface (high priority, 20 hours, ui design)
        - Implement authentication (medium priority, 15 hours, backend auth)
        - Build core features (urgent priority, 40 hours, frontend backend)
        - Testing and QA (medium priority, 25 hours, testing qa)
        """
    }]
)

print(f"Project: {project.project_name}")
print(f"Total hours: {project.total_estimated_hours}")
print(f"High priority tasks: {len(project.get_high_priority_tasks())}")
```

### 2. Union Types and Polymorphism

```python
from typing import Union, Literal
from pydantic import Field, discriminator

class EmailNotification(BaseModel):
    type: Literal["email"] = "email"
    recipient: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    subject: str
    body: str

class SMSNotification(BaseModel):
    type: Literal["sms"] = "sms"
    phone_number: str = Field(..., regex=r'^\+?1?\d{9,15}$')
    message: str = Field(..., max_length=160)

class PushNotification(BaseModel):
    type: Literal["push"] = "push"
    device_id: str
    title: str
    body: str
    badge_count: Optional[int] = None

# Union type with discriminator
Notification = Union[EmailNotification, SMSNotification, PushNotification]

class NotificationRequest(BaseModel):
    """Type-safe notification handling."""
    user_id: str
    notification: Notification = Field(..., discriminator='type')
    urgent: bool = False
    
    def send_notification(self):
        """Type-safe dispatch based on notification type."""
        if isinstance(self.notification, EmailNotification):
            return self._send_email(self.notification)
        elif isinstance(self.notification, SMSNotification):
            return self._send_sms(self.notification)
        elif isinstance(self.notification, PushNotification):
            return self._send_push(self.notification)
    
    def _send_email(self, notification: EmailNotification):
        print(f"Sending email to {notification.recipient}: {notification.subject}")
    
    def _send_sms(self, notification: SMSNotification):
        print(f"Sending SMS to {notification.phone_number}: {notification.message}")
    
    def _send_push(self, notification: PushNotification):
        print(f"Sending push to {notification.device_id}: {notification.title}")

# Extract with automatic type discrimination
request = client.chat.completions.create(
    model="gpt-4",
    response_model=NotificationRequest,
    messages=[{
        "role": "user",
        "content": "Send urgent email to john@example.com with subject 'Meeting Reminder' and body 'Don't forget our 3pm meeting'"
    }]
)

# Type-safe handling
request.send_notification()  # Automatically routes to correct method
```

### 3. Advanced Validation with Custom Validators

```python
from pydantic import validator, ValidationError
from typing import ClassVar
import requests
from urllib.parse import urlparse

class WebsiteAnalysis(BaseModel):
    """Advanced validation for web analysis."""
    url: str
    title: str
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    has_ssl: bool = False
    response_time_ms: Optional[int] = None
    
    # Class-level validation config
    validate_all: ClassVar[bool] = True
    
    @validator('url')
    def validate_url(cls, v):
        """Validate and normalize URL."""
        parsed = urlparse(v)
        if not parsed.scheme:
            v = f"https://{v}"
            parsed = urlparse(v)
        
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('URL must use http or https')
        
        if not parsed.netloc:
            raise ValueError('Invalid URL format')
            
        return v
    
    @validator('has_ssl', pre=False, always=True)
    def check_ssl(cls, v, values):
        """Automatically determine SSL status from URL."""
        url = values.get('url', '')
        return url.startswith('https://')
    
    @validator('keywords')
    def process_keywords(cls, v):
        """Clean and deduplicate keywords."""
        if not v:
            return []
        
        # Clean, deduplicate, and sort
        cleaned = list(set(keyword.lower().strip() for keyword in v if keyword.strip()))
        return sorted(cleaned)
    
    @validator('response_time_ms')
    def validate_response_time(cls, v):
        """Ensure reasonable response time values."""
        if v is not None and (v < 0 or v > 30000):  # 30 second max
            raise ValueError('Response time must be between 0-30000ms')
        return v

# Extract with comprehensive validation
analysis = client.chat.completions.create(
    model="gpt-4",
    response_model=WebsiteAnalysis,
    messages=[{
        "role": "user",
        "content": """
        Analyze this website: example.com
        Title: Example Domain
        Description: This domain is for use in illustrative examples
        Keywords: example, domain, illustration, demo
        Response time: 250ms
        """
    }]
)

print(f"Analyzed: {analysis.url}")
print(f"SSL: {analysis.has_ssl}")
print(f"Keywords: {', '.join(analysis.keywords)}")
```

## Error Handling and Recovery

### 1. Graceful Degradation

```python
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class ExtractedData(BaseModel):
    """Data extraction with confidence scoring."""
    primary_data: Any  # Successfully extracted data
    fallback_data: Optional[Any] = None  # Partial extraction
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_notes: Optional[str] = None

class SafeExtractor:
    """Type-safe extractor with fallback strategies."""
    
    def __init__(self, client):
        self.client = client
    
    def extract_with_fallback(
        self, 
        content: str, 
        primary_model: BaseModel, 
        fallback_model: Optional[BaseModel] = None
    ) -> ExtractedData:
        """Extract with automatic fallback on failure."""
        
        # Try primary extraction
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=primary_model,
                messages=[{"role": "user", "content": content}],
                max_retries=2
            )
            
            return ExtractedData(
                primary_data=result,
                extraction_confidence=0.95,
                extraction_notes="Primary extraction successful"
            )
            
        except ValidationError as e:
            logger.warning(f"Primary extraction failed: {e}")
            
            # Try fallback model
            if fallback_model:
                try:
                    fallback_result = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Faster, more lenient model
                        response_model=fallback_model,
                        messages=[{"role": "user", "content": content}],
                        max_retries=1
                    )
                    
                    return ExtractedData(
                        primary_data=None,
                        fallback_data=fallback_result,
                        extraction_confidence=0.7,
                        extraction_notes=f"Fallback extraction used: {str(e)}"
                    )
                    
                except ValidationError as fallback_error:
                    logger.error(f"Fallback extraction also failed: {fallback_error}")
            
            # Return empty result with error info
            return ExtractedData(
                primary_data=None,
                extraction_confidence=0.0,
                extraction_notes=f"All extraction attempts failed: {str(e)}"
            )

# Define primary and fallback models
class DetailedContact(BaseModel):
    name: str
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    phone: str = Field(..., regex=r'^\+?1?\d{9,15}$')
    company: str
    title: str

class BasicContact(BaseModel):
    name: str
    contact_info: str  # More lenient - any contact info

# Use with fallback
extractor = SafeExtractor(client)

result = extractor.extract_with_fallback(
    content="John Smith, john@example.com, Acme Corp",
    primary_model=DetailedContact,
    fallback_model=BasicContact
)

if result.primary_data:
    print(f"Full extraction: {result.primary_data}")
elif result.fallback_data:
    print(f"Partial extraction: {result.fallback_data}")
else:
    print(f"Extraction failed: {result.extraction_notes}")
```

### 2. Retry Strategies

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import instructor

class RetryableExtractor:
    """Extractor with advanced retry logic."""
    
    def __init__(self, client):
        self.client = client
    
    @retry(
        retry=retry_if_exception_type(ValidationError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def extract_with_custom_retry(self, content: str, model: BaseModel):
        """Custom retry logic for complex validations."""
        return self.client.chat.completions.create(
            model="gpt-4",
            response_model=model,
            messages=[{"role": "user", "content": content}]
        )
    
    async def extract_batch_with_recovery(
        self, 
        contents: List[str], 
        model: BaseModel,
        max_concurrent: int = 5
    ) -> List[ExtractedData]:
        """Batch processing with individual error recovery."""
        import asyncio
        from instructor import AsyncInstructor
        
        async_client = AsyncInstructor.from_openai(AsyncOpenAI())
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_single(content: str) -> ExtractedData:
            async with semaphore:
                try:
                    result = await async_client.chat.completions.create(
                        model="gpt-4",
                        response_model=model,
                        messages=[{"role": "user", "content": content}]
                    )
                    return ExtractedData(
                        primary_data=result,
                        extraction_confidence=0.95
                    )
                except Exception as e:
                    return ExtractedData(
                        primary_data=None,
                        extraction_confidence=0.0,
                        extraction_notes=str(e)
                    )
        
        tasks = [extract_single(content) for content in contents]
        return await asyncio.gather(*tasks)

# Usage
extractor = RetryableExtractor(client)

# Single extraction with retry
contact = extractor.extract_with_custom_retry(
    "Extract: Jane Doe, jane@example.com, 555-123-4567",
    DetailedContact
)

# Batch processing with error recovery
contents = [
    "John Smith, john@example.com, 555-111-2222",
    "Invalid data here",
    "Mary Johnson, mary@example.com, 555-333-4444"
]

results = asyncio.run(extractor.extract_batch_with_recovery(contents, DetailedContact))

successful_extractions = [r for r in results if r.primary_data]
failed_extractions = [r for r in results if not r.primary_data]

print(f"Successful: {len(successful_extractions)}")
print(f"Failed: {len(failed_extractions)}")
```

## Streaming and Real-Time Validation

### 1. Streaming Structured Outputs

```python
from instructor import Partial
import json

class ProgressiveAnalysis(BaseModel):
    """Model that can be built progressively."""
    title: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    
    def is_complete(self) -> bool:
        """Check if analysis is complete enough to use."""
        return (
            self.title is not None 
            and self.summary is not None 
            and len(self.key_points) >= 3
            and self.sentiment is not None
        )
    
    def completeness_score(self) -> float:
        """Calculate how complete the analysis is (0-1)."""
        checks = [
            self.title is not None,
            self.summary is not None,
            len(self.key_points) >= 3,
            self.sentiment is not None,
            self.confidence is not None
        ]
        return sum(checks) / len(checks)

# Stream partial results
for partial_analysis in client.chat.completions.create_partial(
    model="gpt-4",
    response_model=ProgressiveAnalysis,
    messages=[{
        "role": "user",
        "content": "Analyze this article: [long article text...]"
    }]
):
    # Update UI in real-time as fields are populated
    completeness = partial_analysis.completeness_score()
    
    print(f"Progress: {completeness:.1%}")
    
    if partial_analysis.title:
        print(f"Title: {partial_analysis.title}")
    
    if partial_analysis.key_points:
        print(f"Key points so far: {len(partial_analysis.key_points)}")
    
    # Use partial results when complete enough
    if partial_analysis.is_complete():
        print("Analysis complete enough for use!")
        break
```

### 2. Real-Time Data Validation

```python
from typing import Dict, Any
import asyncio
from datetime import datetime

class LiveDataValidator:
    """Real-time data validation system."""
    
    def __init__(self, client):
        self.client = client
        self.validation_cache: Dict[str, Any] = {}
    
    async def validate_stream(
        self, 
        data_stream: AsyncIterator[str], 
        validation_model: BaseModel
    ) -> AsyncIterator[ExtractedData]:
        """Validate streaming data in real-time."""
        
        buffer = ""
        
        async for chunk in data_stream:
            buffer += chunk
            
            # Try validation on accumulated data
            try:
                # Attempt partial parsing
                partial_result = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Fast model for real-time
                    response_model=Partial[validation_model],
                    messages=[{
                        "role": "user",
                        "content": f"Extract what you can from: {buffer}"
                    }]
                )
                
                yield ExtractedData(
                    primary_data=partial_result,
                    extraction_confidence=0.8,
                    extraction_notes=f"Partial validation at {datetime.now()}"
                )
                
            except ValidationError:
                # Continue accumulating data
                continue
    
    def validate_with_cache(self, content: str, model: BaseModel) -> ExtractedData:
        """Validate with caching for repeated patterns."""
        
        # Simple hash for caching
        content_hash = hash(content)
        
        if content_hash in self.validation_cache:
            cached_result = self.validation_cache[content_hash]
            return ExtractedData(
                primary_data=cached_result,
                extraction_confidence=0.95,
                extraction_notes="Retrieved from cache"
            )
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=model,
                messages=[{"role": "user", "content": content}]
            )
            
            # Cache successful results
            self.validation_cache[content_hash] = result
            
            return ExtractedData(
                primary_data=result,
                extraction_confidence=0.95,
                extraction_notes="Validated and cached"
            )
            
        except ValidationError as e:
            return ExtractedData(
                primary_data=None,
                extraction_confidence=0.0,
                extraction_notes=f"Validation failed: {str(e)}"
            )

# Usage
validator = LiveDataValidator(client)

# Cache-enabled validation
result = validator.validate_with_cache(
    "Process this data: John, 25, Engineer",
    DetailedContact
)

print(f"Validation result: {result.extraction_notes}")
```

## Testing Type-Safe AI Applications

### 1. Unit Testing with Pydantic

```python
import pytest
from pydantic import ValidationError

class TestContactValidation:
    """Test suite for contact validation."""
    
    def test_valid_contact_creation(self):
        """Test creating valid contact."""
        contact = Contact(
            name="John Doe",
            email="john@example.com",
            phone="+1-555-123-4567",
            confidence=0.95
        )
        
        assert contact.name == "John Doe"
        assert contact.email == "john@example.com"
        assert contact.confidence == 0.95
    
    def test_invalid_email_validation(self):
        """Test email validation."""
        with pytest.raises(ValidationError) as exc_info:
            Contact(
                name="John Doe",
                email="invalid-email",
                confidence=0.95
            )
        
        assert "email" in str(exc_info.value)
    
    def test_confidence_bounds(self):
        """Test confidence score validation."""
        with pytest.raises(ValidationError):
            Contact(name="John", confidence=1.5)  # > 1.0
        
        with pytest.raises(ValidationError):
            Contact(name="John", confidence=-0.1)  # < 0.0
    
    def test_phone_normalization(self):
        """Test phone number normalization."""
        contact = Contact(
            name="John",
            phone="5551234567",  # Should be normalized
            confidence=0.9
        )
        
        assert "+1-555-123-4567" in contact.phone

class TestExtractorIntegration:
    """Integration tests for the full extraction pipeline."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock client for testing."""
        class MockClient:
            def chat_completions_create(self, **kwargs):
                # Return mock structured response
                return Contact(
                    name="Test User",
                    email="test@example.com",
                    confidence=0.9
                )
        
        return MockClient()
    
    def test_successful_extraction(self, mock_client):
        """Test successful extraction flow."""
        extractor = SafeExtractor(mock_client)
        
        result = extractor.extract_with_fallback(
            "Extract: Test User, test@example.com",
            Contact,
            BasicContact
        )
        
        assert result.primary_data is not None
        assert result.extraction_confidence > 0.8
    
    def test_extraction_fallback(self, mock_client):
        """Test fallback mechanism."""
        # Configure mock to fail primary, succeed fallback
        def mock_extract(**kwargs):
            if kwargs['response_model'] == Contact:
                raise ValidationError("Primary failed", Contact)
            return BasicContact(name="Test", contact_info="test@example.com")
        
        mock_client.chat_completions_create = mock_extract
        extractor = SafeExtractor(mock_client)
        
        result = extractor.extract_with_fallback(
            "Extract: Test User",
            Contact,
            BasicContact
        )
        
        assert result.primary_data is None
        assert result.fallback_data is not None
        assert result.extraction_confidence < 0.8

# Property-based testing
from hypothesis import given, strategies as st

class TestPropertyBasedValidation:
    """Property-based tests using Hypothesis."""
    
    @given(
        name=st.text(min_size=1, max_size=100),
        confidence=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_contact_properties(self, name, confidence):
        """Test that any valid inputs create valid contacts."""
        try:
            contact = Contact(name=name, confidence=confidence)
            
            # Properties that should always hold
            assert len(contact.name) >= 1
            assert 0.0 <= contact.confidence <= 1.0
            assert isinstance(contact, Contact)
            
        except ValidationError:
            # Some inputs may fail validation, that's ok
            pass
```

### 2. Integration Testing

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

class TestProductionWorkflow:
    """Test complete production workflows."""
    
    @pytest.fixture
    def production_client(self):
        """Production-like client setup."""
        return instructor.from_openai(OpenAI(api_key="test-key"))
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, production_client):
        """Test batch processing workflow."""
        
        # Mock the actual API calls
        production_client.chat.completions.create = AsyncMock(
            return_value=Contact(
                name="Test User",
                email="test@example.com",
                confidence=0.9
            )
        )
        
        extractor = RetryableExtractor(production_client)
        
        test_data = [
            "John Doe, john@example.com",
            "Jane Smith, jane@example.com",
            "Bob Wilson, bob@example.com"
        ]
        
        results = await extractor.extract_batch_with_recovery(
            test_data, 
            Contact,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all(r.primary_data is not None for r in results)
    
    def test_error_propagation(self, production_client):
        """Test that errors are properly handled and logged."""
        
        # Configure to always fail
        production_client.chat.completions.create = Mock(
            side_effect=ValidationError("Test error", Contact)
        )
        
        extractor = SafeExtractor(production_client)
        
        result = extractor.extract_with_fallback(
            "Test input",
            Contact,
            None  # No fallback
        )
        
        assert result.primary_data is None
        assert result.extraction_confidence == 0.0
        assert "Test error" in result.extraction_notes

# Performance testing
class TestPerformance:
    """Performance and load testing."""
    
    def test_validation_performance(self):
        """Test validation performance under load."""
        import time
        
        start_time = time.time()
        
        # Create many contacts
        for i in range(1000):
            contact = Contact(
                name=f"User {i}",
                email=f"user{i}@example.com",
                confidence=0.9
            )
            assert contact.name == f"User {i}"
        
        elapsed = time.time() - start_time
        
        # Should validate 1000 contacts in under 1 second
        assert elapsed < 1.0
        
        print(f"Validated 1000 contacts in {elapsed:.3f}s")
```

## Production Deployment Patterns

### 1. Configuration and Environment Management

```python
from pydantic import BaseSettings, Field
from typing import Optional
import os

class AIApplicationSettings(BaseSettings):
    """Type-safe application settings."""
    
    # API Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    default_model: str = Field("gpt-4", env="DEFAULT_MODEL")
    
    # Application Settings
    max_retries: int = Field(3, env="MAX_RETRIES")
    timeout_seconds: int = Field(30, env="TIMEOUT_SECONDS")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    
    # Validation Settings
    strict_validation: bool = Field(True, env="STRICT_VALIDATION")
    confidence_threshold: float = Field(0.8, env="CONFIDENCE_THRESHOLD")
    
    # Monitoring
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class ProductionAIService:
    """Production-ready AI service with full type safety."""
    
    def __init__(self, settings: AIApplicationSettings):
        self.settings = settings
        self.client = self._setup_client()
        self.metrics = self._setup_metrics()
    
    def _setup_client(self):
        """Setup AI client with configuration."""
        if self.settings.openai_api_key:
            from openai import OpenAI
            return instructor.from_openai(
                OpenAI(
                    api_key=self.settings.openai_api_key,
                    timeout=self.settings.timeout_seconds
                )
            )
        elif self.settings.anthropic_api_key:
            from anthropic import Anthropic
            return instructor.from_anthropic(
                Anthropic(api_key=self.settings.anthropic_api_key)
            )
        else:
            raise ValueError("No API key provided")
    
    def _setup_metrics(self):
        """Setup metrics collection."""
        if self.settings.enable_metrics:
            import prometheus_client
            return {
                'extraction_requests': prometheus_client.Counter(
                    'ai_extraction_requests_total',
                    'Total extraction requests'
                ),
                'extraction_duration': prometheus_client.Histogram(
                    'ai_extraction_duration_seconds',
                    'Time spent on extractions'
                ),
                'validation_errors': prometheus_client.Counter(
                    'ai_validation_errors_total',
                    'Total validation errors'
                )
            }
        return {}
    
    def extract_with_monitoring(
        self, 
        content: str, 
        model: BaseModel,
        context: Optional[Dict] = None
    ) -> ExtractedData:
        """Extract with full monitoring and error handling."""
        
        if self.settings.enable_metrics:
            self.metrics['extraction_requests'].inc()
        
        start_time = time.time()
        
        try:
            result = self.client.chat.completions.create(
                model=self.settings.default_model,
                response_model=model,
                messages=[{"role": "user", "content": content}],
                max_retries=self.settings.max_retries
            )
            
            # Validate confidence threshold
            if hasattr(result, 'confidence'):
                if result.confidence < self.settings.confidence_threshold:
                    logger.warning(f"Low confidence result: {result.confidence}")
            
            duration = time.time() - start_time
            
            if self.settings.enable_metrics:
                self.metrics['extraction_duration'].observe(duration)
            
            return ExtractedData(
                primary_data=result,
                extraction_confidence=getattr(result, 'confidence', 0.95),
                extraction_notes=f"Extracted in {duration:.2f}s"
            )
            
        except ValidationError as e:
            if self.settings.enable_metrics:
                self.metrics['validation_errors'].inc()
            
            logger.error(f"Validation error: {e}")
            
            return ExtractedData(
                primary_data=None,
                extraction_confidence=0.0,
                extraction_notes=f"Validation failed: {str(e)}"
            )

# Usage in production
settings = AIApplicationSettings()
service = ProductionAIService(settings)

# Type-safe extraction with monitoring
result = service.extract_with_monitoring(
    "Extract: John Doe, john@example.com",
    Contact,
    context={"user_id": "12345", "source": "web_form"}
)
```

### 2. Logging and Observability

```python
import logging
import structlog
from typing import Dict, Any
import json

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ObservableExtractor:
    """Extractor with full observability."""
    
    def __init__(self, client, service_name: str = "ai-extractor"):
        self.client = client
        self.service_name = service_name
        self.logger = logger.bind(service=service_name)
    
    def extract_with_observability(
        self,
        content: str,
        model: BaseModel,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ExtractedData:
        """Extract with comprehensive logging and tracing."""
        
        request_id = request_id or f"req_{int(time.time())}"
        
        # Structured logging context
        log_context = {
            "request_id": request_id,
            "user_id": user_id,
            "model_name": model.__name__,
            "content_length": len(content),
            "operation": "extract"
        }
        
        request_logger = self.logger.bind(**log_context)
        request_logger.info("Starting extraction")
        
        start_time = time.time()
        
        try:
            # Log input (be careful with PII)
            if len(content) < 500:  # Only log short content
                request_logger.debug("Processing content", content_preview=content[:100])
            
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=model,
                messages=[{"role": "user", "content": content}]
            )
            
            duration = time.time() - start_time
            
            # Log successful extraction
            request_logger.info(
                "Extraction completed successfully",
                duration_seconds=duration,
                result_type=type(result).__name__
            )
            
            return ExtractedData(
                primary_data=result,
                extraction_confidence=0.95,
                extraction_notes=f"Completed in {duration:.2f}s"
            )
            
        except ValidationError as e:
            duration = time.time() - start_time
            
            request_logger.error(
                "Validation error during extraction",
                error=str(e),
                duration_seconds=duration,
                error_type="ValidationError"
            )
            
            return ExtractedData(
                primary_data=None,
                extraction_confidence=0.0,
                extraction_notes=f"Validation error: {str(e)}"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            request_logger.error(
                "Unexpected error during extraction",
                error=str(e),
                duration_seconds=duration,
                error_type=type(e).__name__,
                exc_info=True
            )
            
            return ExtractedData(
                primary_data=None,
                extraction_confidence=0.0,
                extraction_notes=f"Unexpected error: {str(e)}"
            )

# Usage with observability
extractor = ObservableExtractor(client, "production-extractor")

result = extractor.extract_with_observability(
    content="Extract: John Doe, john@example.com",
    model=Contact,
    request_id="req_12345",
    user_id="user_67890"
)

# Logs will be structured JSON for easy querying:
# {
#   "event": "Starting extraction",
#   "timestamp": "2025-06-17T10:30:00.123456Z",
#   "level": "info",
#   "service": "production-extractor",
#   "request_id": "req_12345",
#   "user_id": "user_67890",
#   "model_name": "Contact",
#   "content_length": 35,
#   "operation": "extract"
# }
```

## Best Practices Summary

### 1. Model Design
- Use descriptive field names and docstrings
- Add validation constraints (min/max, regex, etc.)
- Include confidence scores for uncertain extractions
- Design for composition and reusability

### 2. Error Handling
- Always handle ValidationError explicitly
- Implement fallback strategies for critical paths
- Use structured logging for debugging
- Include confidence metrics in responses

### 3. Performance
- Cache validation results for repeated patterns
- Use streaming for long-running extractions
- Implement proper async patterns for batch processing
- Monitor extraction latency and success rates

### 4. Testing
- Unit test all validation logic
- Integration test complete workflows
- Use property-based testing for edge cases
- Mock API calls for reliable CI/CD

### 5. Production Deployment
- Use type-safe configuration management
- Implement comprehensive monitoring
- Add structured logging with request tracing
- Plan for graceful degradation

## Conclusion

Type-safe AI applications built with Instructor and Pydantic provide:

- **Reliability**: Guaranteed data structures prevent runtime errors
- **Maintainability**: Clear contracts between components
- **Debuggability**: Structured validation errors and logging
- **Performance**: Efficient validation and caching strategies
- **Scalability**: Async patterns for high-throughput applications

By following these patterns and practices, you can build AI applications that are production-ready, maintainable, and reliable.

## Related Concepts

- [Structured Output from LLMs: The Complete Guide](structured-output-llm-complete-guide.md) - Foundational concepts
- [Models and Response Models](../../concepts/models.md) - Advanced Pydantic patterns
- [Validation and Error Handling](../../concepts/validation.md) - Comprehensive validation strategies

## See Also

- [From Messy JSON to Clean Data Models](messy-json-clean-data-models.md) - Data transformation patterns
- [10 Instructor Patterns That Save Hours](instructor-patterns-save-hours.md) - Advanced techniques
- [Examples Gallery](../../examples/index.md) - Real-world implementations

Start building type-safe AI applications today with [Instructor](https://github.com/jxnl/instructor)!