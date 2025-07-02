"""Async engine for handling LLM calls with concurrency control and batching."""

import asyncio
import os
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from jinja2 import Environment, Template
from pydantic import BaseModel

import instructor

from .types import LLMError, TemplateError, TemplateKwargsType, TemplateType
from .utils import validate_template_kwargs


class AsyncEngine:
    """Async orchestrator for LLM calls with concurrency control."""

    def __init__(self, client: Optional[Any] = None, max_concurrency: int = 32):
        """
        Initialize the async engine.

        Args:
            client: instructor client, defaults to openai/gpt-4o
            max_concurrency: Maximum number of concurrent requests
        """
        self.client = client or instructor.from_provider("openai/gpt-4o")
        self.max_concurrency = int(
            os.getenv("PROMPTFRAME_MAX_CONCURRENCY", max_concurrency)
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.jinja_env = Environment()
        self._template_cache: Dict[str, Template] = {}

    def _get_template(self, template_str: str) -> Template:
        """Get or create a Jinja template with caching."""
        if template_str not in self._template_cache:
            self._template_cache[template_str] = self.jinja_env.from_string(
                template_str
            )
        return self._template_cache[template_str]

    def _render_prompt(
        self,
        template: TemplateType,
        row: pd.Series,
        template_kwargs: TemplateKwargsType,
    ) -> str:
        """Render a prompt for a single row."""
        try:
            if callable(template):
                return template(row)

            # Handle Jinja template
            jinja_template = self._get_template(template)

            # Combine row data with additional template kwargs
            context = dict(row)
            context.update(validate_template_kwargs(template_kwargs, row))

            return jinja_template.render(**context)
        except Exception as e:
            raise TemplateError(f"Failed to render template for row {row.name}: {e}")

    async def _call_llm_single(
        self,
        prompt: str,
        schema: Type[BaseModel],
        llm_model: str,
        **provider_kwargs,
    ) -> BaseModel:
        """Make a single LLM call."""
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=llm_model,
                    response_model=schema,
                    messages=[{"role": "user", "content": prompt}],
                    **provider_kwargs,
                )
                return response
            except Exception as e:
                raise LLMError(f"LLM call failed: {e}")

    async def _call_llm_batch(
        self,
        prompts: List[str],
        schema: Type[BaseModel],
        llm_model: str,
        **provider_kwargs,
    ) -> List[BaseModel]:
        """Make a batched LLM call."""
        # For batching, we need to modify the schema to expect a list
        from pydantic import create_model

        # Create a wrapper model that expects a list of the original schema
        ListSchema = create_model(f"Batch{schema.__name__}", items=(List[schema], ...))

        # Combine prompts with separators
        combined_prompt = "\n---\n".join(prompts)
        batch_prompt = f"""
Please process each of the following {len(prompts)} items separately and return a JSON array with {len(prompts)} results:

{combined_prompt}

Return the results as a JSON array where each element corresponds to one input item in the same order.
"""

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=llm_model,
                    response_model=ListSchema,
                    messages=[{"role": "user", "content": batch_prompt}],
                    **provider_kwargs,
                )
                return response.items
            except Exception:
                # Fallback to individual calls if batching fails
                return await asyncio.gather(
                    *[
                        self._call_llm_single(prompt, schema, llm_model, **provider_kwargs)
                        for prompt in prompts
                    ]
                )

    async def process_dataframe(
        self,
        df: pd.DataFrame,
        template: TemplateType,
        schema: Type[BaseModel],
        llm_model: str = "openai/gpt-4o",
        template_kwargs: TemplateKwargsType = None,
        batch_size: int = 8,
        progress: bool = True,
        **provider_kwargs,
    ) -> List[BaseModel]:
        """
        Process a DataFrame asynchronously with LLM calls.

        Args:
            df: DataFrame to process
            template: Jinja template string or callable
            schema: Pydantic model for response validation
            llm_model: Model to use for LLM calls
            template_kwargs: Additional template variables
            batch_size: Number of rows to batch together (1 = no batching)
            progress: Whether to show progress
            **provider_kwargs: Additional arguments for LLM provider

        Returns:
            List of Pydantic model instances
        """
        if progress:
            try:
                from tqdm.asyncio import tqdm

                progress_bar = tqdm(total=len(df), desc="Processing rows")
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None

        # Render all prompts first
        prompts = []
        for _, row in df.iterrows():
            prompt = self._render_prompt(template, row, template_kwargs)
            prompts.append(prompt)

        results = []

        if batch_size > 1:
            # Process in batches
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                batch_results = await self._call_llm_batch(
                    batch_prompts, schema, llm_model, **provider_kwargs
                )
                results.extend(batch_results)

                if progress_bar:
                    progress_bar.update(len(batch_prompts))
        else:
            # Process individually with concurrency
            tasks = [
                self._call_llm_single(prompt, schema, llm_model, **provider_kwargs)
                for prompt in prompts
            ]

            if progress_bar:
                # Use asyncio.gather with progress updates
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    progress_bar.update(1)
                # Sort results back to original order
                # Note: This is a simplified approach; in practice, you'd want to maintain order
                # For now, we'll use gather which maintains order
                results = await asyncio.gather(*tasks)
            else:
                results = await asyncio.gather(*tasks)

        if progress_bar:
            progress_bar.close()

        return results