"""Unit tests for promptframe."""

import pytest
import pandas as pd
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Import the modules we want to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from promptframe import PromptFrame, PromptFrameError
from promptframe.utils import expand_pydantic_to_columns, merge_columns_to_dataframe


class MockAnalysis(BaseModel):
    """Mock schema for testing."""
    summary: str = Field(description="Summary of the text")
    sentiment: str = Field(description="Sentiment analysis")
    tags: list[str] = Field(description="Tags for the content")


class TestPromptFrame:
    """Test cases for PromptFrame class."""
    
    def test_init(self):
        """Test PromptFrame initialization."""
        df = pd.DataFrame({"text": ["hello", "world"]})
        pf = PromptFrame(df)
        
        assert pf.shape == (2, 1)
        assert len(pf) == 2
        assert not pf.has_errors()
        assert list(pf.columns) == ["text"]
    
    def test_init_with_copy(self):
        """Test that PromptFrame creates a copy of the input DataFrame."""
        df = pd.DataFrame({"text": ["hello", "world"]})
        pf = PromptFrame(df)
        
        # Modify original DataFrame
        df.loc[0, "text"] = "modified"
        
        # PromptFrame should have original data
        assert pf.to_df().loc[0, "text"] == "hello"
    
    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        df = pd.DataFrame({"text": []})
        pf = PromptFrame(df)
        
        # Should not fail with empty DataFrame
        result = pf.map_prompt(
            name="test",
            template="{{text}}",
            schema=MockAnalysis
        )
        
        assert result is pf  # Should return self
        assert pf.to_df().equals(df)  # Should be unchanged
    
    def test_validation_errors(self):
        """Test input validation."""
        df = pd.DataFrame({"text": ["hello"]})
        pf = PromptFrame(df)
        
        # Test invalid name
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            pf.map_prompt(name="", template="{{text}}", schema=MockAnalysis)
        
        # Test invalid schema
        with pytest.raises(ValueError, match="schema must be a Pydantic BaseModel subclass"):
            pf.map_prompt(name="test", template="{{text}}", schema=str)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_expand_pydantic_to_columns(self):
        """Test expanding Pydantic models to columns."""
        # Create mock model instances
        models = [
            MockAnalysis(summary="Test 1", sentiment="positive", tags=["tag1", "tag2"]),
            MockAnalysis(summary="Test 2", sentiment="negative", tags=["tag3"])
        ]
        
        result = expand_pydantic_to_columns(models, prefix="analysis")
        
        expected = {
            "analysis.summary": ["Test 1", "Test 2"],
            "analysis.sentiment": ["positive", "negative"],
            "analysis.tags": [["tag1", "tag2"], ["tag3"]]
        }
        
        assert result == expected
    
    def test_expand_pydantic_empty_list(self):
        """Test expanding empty list of models."""
        result = expand_pydantic_to_columns([], prefix="test")
        assert result == {}
    
    def test_merge_columns_to_dataframe(self):
        """Test merging new columns into DataFrame."""
        df = pd.DataFrame({"original": [1, 2]})
        new_columns = {
            "new_col1": ["a", "b"],
            "new_col2": [10, 20]
        }
        
        result = merge_columns_to_dataframe(df, new_columns)
        
        expected = pd.DataFrame({
            "original": [1, 2],
            "new_col1": ["a", "b"], 
            "new_col2": [10, 20]
        })
        
        pd.testing.assert_frame_equal(result, expected)
    
    def test_merge_empty_columns(self):
        """Test merging empty columns dictionary."""
        df = pd.DataFrame({"original": [1, 2]})
        result = merge_columns_to_dataframe(df, {})
        
        pd.testing.assert_frame_equal(result, df)


class TestAsyncEngine:
    """Test cases for AsyncEngine (mocked)."""
    
    @pytest.mark.asyncio
    async def test_template_rendering(self):
        """Test template rendering functionality."""
        from promptframe.engine import AsyncEngine
        
        engine = AsyncEngine()
        
        # Test string template
        row = pd.Series({"name": "John", "age": 30})
        template = "Hello {{name}}, you are {{age}} years old"
        
        result = engine._render_prompt(template, row, None)
        assert result == "Hello John, you are 30 years old"
    
    @pytest.mark.asyncio
    async def test_callable_template(self):
        """Test callable template rendering."""
        from promptframe.engine import AsyncEngine
        
        engine = AsyncEngine()
        
        def template_func(row):
            return f"Name: {row['name']}, Age: {row['age']}"
        
        row = pd.Series({"name": "John", "age": 30})
        result = engine._render_prompt(template_func, row, None)
        
        assert result == "Name: John, Age: 30"


class TestIntegration:
    """Integration tests (require mocking LLM calls)."""
    
    @patch('promptframe.engine.instructor.from_provider')
    def test_map_prompt_integration(self, mock_instructor):
        """Test full map_prompt integration with mocked LLM."""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MockAnalysis(
            summary="Test summary",
            sentiment="positive", 
            tags=["test", "mock"]
        )
        
        # Mock async chat.completions.create
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_instructor.return_value = mock_client
        
        # Test data
        df = pd.DataFrame({"text": ["Test text"]})
        pf = PromptFrame(df)
        
        # Run with mocked LLM
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = [mock_response]
            
            result = pf.map_prompt(
                name="analysis",
                template="Analyze: {{text}}",
                schema=MockAnalysis,
                progress=False
            )
        
        assert result is pf  # Should return self
        
        # Check that DataFrame has new columns
        enriched = pf.to_df()
        expected_columns = ["text", "analysis.summary", "analysis.sentiment", "analysis.tags"]
        
        for col in expected_columns:
            assert col in enriched.columns


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for tests."""
    return pd.DataFrame({
        "text": [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ],
        "category": ["review", "review", "review"]
    })


@pytest.fixture
def sample_schema():
    """Fixture providing sample Pydantic schema."""
    return MockAnalysis


if __name__ == "__main__":
    pytest.main([__file__])