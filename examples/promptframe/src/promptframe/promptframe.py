"""
PromptFrame - Simple DataFrame + LLM processing

One file, minimal dependencies, easy to understand, fast async processing with lazy evaluation.
"""

import asyncio
import pandas as pd
from typing import Optional, Any
from pydantic import BaseModel
import instructor
from tqdm.asyncio import tqdm


class PromptOperation:
    """Represents a queued prompt operation."""

    def __init__(
        self, name: str, response_model: type[BaseModel], template: Optional[str]
    ):
        self.name = name
        self.response_model = response_model
        self.template = template


class PromptFrame:
    """Simple DataFrame wrapper for LLM processing with async support and lazy evaluation."""

    def __init__(
        self,
        df: pd.DataFrame,
        client: Any,
        max_concurrency: int = 10,
        batch_size: Optional[int] = None,
    ):
        self.df = df.copy()
        self.client = client
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self._operations: list[PromptOperation] = []

    def map_prompt(
        self, name: str, response_model: type[BaseModel], template: Optional[str] = None
    ) -> "PromptFrame":
        """
        Queue an LLM operation to apply to each row (lazy evaluation).

        Args:
            name: Column prefix for results (e.g. "analysis" -> "analysis.summary")
            response_model: Pydantic model defining the output structure
            template: Custom template, or None for auto XML template

        Returns:
            Self for chaining (operations are queued, not executed)
        """
        operation = PromptOperation(name, response_model, template)
        self._operations.append(operation)
        return self

    async def collect(self) -> pd.DataFrame:
        """
        Execute all queued operations and return the enriched DataFrame.

        Returns:
            DataFrame with all LLM results added as new columns
        """
        current_df = self.df.copy()

        for operation in self._operations:
            # Auto-generate XML template if none provided
            if operation.template is None:
                operation.template = self._make_xml_template(current_df)

            # Run async processing with progress bar
            results = await self._process_async(
                current_df,
                operation.template,
                operation.response_model,
                self.max_concurrency,
                operation.name,
                self.batch_size,
            )

            # Add results as new columns
            current_df = self._add_results(current_df, results, operation.name)

        return current_df

    async def _process_async(
        self,
        df: pd.DataFrame,
        template: str,
        response_model: type[BaseModel],
        max_concurrency: int,
        operation_name: str,
        batch_size: Optional[int] = None,
    ) -> list:
        """Process all rows concurrently with controlled concurrency and progress tracking."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_row(row):
            async with semaphore:
                prompt = self._render_template(template, row)
                return await self.client.create(
                    response_model=response_model,
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )

        # Process in batches if batch_size is specified
        if batch_size is not None and len(df) > batch_size:
            return await self._process_in_batches(
                df, process_row, operation_name, batch_size
            )

        # Create tasks for all rows
        tasks = [process_row(row) for _, row in df.iterrows()]

        # Run all tasks concurrently with progress bar
        return await tqdm.gather(
            *tasks, desc=f"Processing {operation_name}", total=len(tasks)
        )

    async def _process_in_batches(
        self, df: pd.DataFrame, process_row_fn, operation_name: str, batch_size: int
    ) -> list:
        """Process DataFrame in batches to manage memory and API rate limits."""
        all_results = []
        total_rows = len(df)

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            # Create tasks for this batch
            batch_tasks = [process_row_fn(row) for _, row in batch_df.iterrows()]

            # Process batch with progress tracking
            batch_results = await tqdm.gather(
                *batch_tasks,
                desc=f"Processing {operation_name} (batch {start_idx // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size})",
                total=len(batch_tasks),
            )

            all_results.extend(batch_results)

            # Optional: Add a small delay between batches to be nice to APIs
            if end_idx < total_rows:
                await asyncio.sleep(0.1)

        return all_results

    def _make_xml_template(self, df: pd.DataFrame) -> str:
        """Generate XML template for original columns only (not generated ones)."""
        xml_parts = []
        for col in df.columns:
            # Skip columns that were added by previous operations (contain '.')
            if "." not in col:
                # Clean column name for XML
                clean_col = col.replace(" ", "_").replace("-", "_").replace(".", "_")
                xml_parts.append(f"<{clean_col}>{{{{ {col} }}}}</{clean_col}>")

        return (
            "This is a row of a database, please extract the following information:\n"
            + "\n".join(xml_parts)
        )

    def _render_template(self, template: str, row: pd.Series) -> str:
        """Render Jinja template with row data."""
        from jinja2 import Template

        jinja_template = Template(template)
        return jinja_template.render(**dict(row))

    def _add_results(
        self, df: pd.DataFrame, results: list, prefix: str
    ) -> pd.DataFrame:
        """Add Pydantic results as new DataFrame columns."""
        if not results:
            return df

        # Create a copy to avoid modifying the original
        new_df = df.copy()

        # Get all fields from first result
        first_result = results[0].model_dump()

        # Create new columns for each field
        for field_name, _ in first_result.items():
            col_name = f"{prefix}.{field_name}"
            values = [getattr(result, field_name) for result in results]
            new_df[col_name] = values

        return new_df

    def to_df(self) -> pd.DataFrame:
        """Return the current DataFrame (without executing queued operations)."""
        return self.df.copy()


# Simple usage example
if __name__ == "__main__":

    async def main():
        # Example schema
        class Analysis(BaseModel):
            sentiment: str
            summary: str

        # Example data
        df = pd.DataFrame(
            {
                "text": [
                    "I love this product!",
                    "This is terrible.",
                    "It's okay I guess.",
                ]
            }
        )

        client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

        # Process (requires OPENAI_API_KEY)
        # Operations are queued, not executed immediately
        result = await (
            PromptFrame(df, client, max_concurrency=5, batch_size=2)
            .map_prompt("analysis", Analysis)
            .collect()
        )  # This triggers execution with progress bar

        print(result)

    asyncio.run(main())
