"""Core PromptFrame implementation."""

import asyncio
from typing import Any, List, Optional, Type

import pandas as pd

from .engine import AsyncEngine
from .types import PromptFrameError, TemplateType, TemplateKwargsType
from .utils import (
    expand_pydantic_to_columns,
    generate_default_xml_template,
    merge_columns_to_dataframe,
)

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:  # type: ignore
        pass


class PromptFrame:
    """
    A lightweight bridge between Pandas rows and instructor-powered,
    Jinja-templated, structured-LLM extraction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        client: Optional[Any] = None,
        max_concurrency: int = 32,
    ):
        """
        Initialize PromptFrame with a DataFrame.

        Args:
            df: Input DataFrame to enrich
            client: instructor client, defaults to openai/gpt-4o via instructor.from_provider
            max_concurrency: Maximum concurrent LLM requests
        """
        self.df = df.copy()  # Work with a copy to avoid modifying original
        self.errors: List[PromptFrameError] = []
        self.engine = AsyncEngine(client=client, max_concurrency=max_concurrency)

    def map_prompt(
        self,
        name: str,
        schema: Type[BaseModel],
        template: Optional[TemplateType] = None,
        *,
        llm_model: str = "openai/gpt-4o",
        template_kwargs: TemplateKwargsType = None,
        batch_size: int = 8,
        progress: bool = True,
        **provider_kwargs,
    ) -> "PromptFrame":
        """
        Apply LLM processing to DataFrame rows with structured output.

        Args:
            name: Prefix for new columns (e.g., "analysis" -> "analysis.summary")
            schema: Pydantic model class for structured output validation
            template: Jinja template string or callable (if None, uses default XML template)
            llm_model: Model identifier (default: "openai/gpt-4o")
            template_kwargs: Additional variables for template rendering (dict or callable)
            batch_size: Number of rows to batch together (1 = individual calls)
            progress: Whether to show progress bar
            **provider_kwargs: Additional arguments for LLM provider (temperature, top_p, etc.)

        Returns:
            Self for method chaining

        Raises:
            PromptFrameError: If processing fails

        Note:
            If no template is provided, a default XML template will be generated that wraps
            each column in XML tags like:
            ```
            This is a row of a database, please extract the following information:
            <column_name>{{ column_name }}</column_name>
            ```
        """
        # Validate inputs
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        try:
            if not issubclass(schema, BaseModel):
                raise ValueError("schema must be a Pydantic BaseModel subclass")
        except (TypeError, NameError):
            raise ValueError("schema must be a Pydantic BaseModel subclass")

        if self.df.empty:
            return self

        # Use default XML template if none provided
        if template is None:
            template = generate_default_xml_template(self.df)
            if progress:
                print(f"Using default XML template for columns: {list(self.df.columns)}")

        try:
            # Run async processing
            results = asyncio.run(
                self.engine.process_dataframe(
                    df=self.df,
                    template=template,
                    schema=schema,
                    llm_model=llm_model,
                    template_kwargs=template_kwargs,
                    batch_size=batch_size,
                    progress=progress,
                    **provider_kwargs,
                )
            )

            # Expand Pydantic models to columns
            new_columns = expand_pydantic_to_columns(results, prefix=name)

            # Merge new columns into DataFrame
            self.df = merge_columns_to_dataframe(self.df, new_columns)

        except Exception as e:
            error = PromptFrameError(f"Failed to process prompt '{name}': {e}")
            self.errors.append(error)
            raise error

        return self

    def to_df(self) -> pd.DataFrame:
        """
        Return the enriched DataFrame.

        Returns:
            DataFrame with new LLM-derived columns
        """
        return self.df.copy()

    def get_errors(self) -> List[PromptFrameError]:
        """
        Get list of errors that occurred during processing.

        Returns:
            List of PromptFrameError instances
        """
        return self.errors.copy()

    def has_errors(self) -> bool:
        """
        Check if any errors occurred during processing.

        Returns:
            True if errors exist, False otherwise
        """
        return len(self.errors) > 0

    def clear_errors(self) -> "PromptFrame":
        """
        Clear the error list.

        Returns:
            Self for method chaining
        """
        self.errors.clear()
        return self

    def __repr__(self) -> str:
        """String representation of PromptFrame."""
        return f"PromptFrame(shape={self.df.shape}, errors={len(self.errors)})"

    def __len__(self) -> int:
        """Return number of rows in DataFrame."""
        return len(self.df)

    @property
    def shape(self) -> tuple:
        """Return shape of the DataFrame."""
        return self.df.shape

    @property
    def columns(self) -> pd.Index:
        """Return DataFrame columns."""
        return self.df.columns