"""
PromptFrame - Simple DataFrame + LLM processing

One file, minimal dependencies, easy to understand.
"""

import pandas as pd
from typing import Type, Optional, Any
from pydantic import BaseModel
import instructor


class PromptFrame:
    """Simple DataFrame wrapper for LLM processing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.client = instructor.from_provider("openai/gpt-4o")
    
    def map_prompt(
        self, 
        name: str, 
        schema: Type[BaseModel], 
        template: Optional[str] = None
    ) -> 'PromptFrame':
        """
        Apply LLM to each row and add structured results as new columns.
        
        Args:
            name: Column prefix for results (e.g. "analysis" -> "analysis.summary")
            schema: Pydantic model defining the output structure
            template: Custom template, or None for auto XML template
            
        Returns:
            Self for chaining
        """
        # Auto-generate XML template if none provided
        if template is None:
            template = self._make_xml_template()
        
        # Process each row
        results = []
        for _, row in self.df.iterrows():
            prompt = self._render_template(template, row)
            result = self.client.chat.completions.create(
                model="openai/gpt-4o",
                response_model=schema,
                messages=[{"role": "user", "content": prompt}]
            )
            results.append(result)
        
        # Add results as new columns
        self._add_results(results, name)
        return self
    
    def _make_xml_template(self) -> str:
        """Generate XML template for all columns."""
        xml_parts = []
        for col in self.df.columns:
            # Clean column name for XML
            clean_col = col.replace(" ", "_").replace("-", "_").replace(".", "_")
            xml_parts.append(f"<{clean_col}>{{{{ {col} }}}}</{clean_col}>")
        
        return (
            "This is a row of a database, please extract the following information:\n" + 
            "\n".join(xml_parts)
        )
    
    def _render_template(self, template: str, row: pd.Series) -> str:
        """Render Jinja template with row data."""
        from jinja2 import Template
        jinja_template = Template(template)
        return jinja_template.render(**dict(row))
    
    def _add_results(self, results: list, prefix: str):
        """Add Pydantic results as new DataFrame columns."""
        if not results:
            return
            
        # Get all fields from first result
        first_result = results[0].model_dump()
        
        # Create new columns for each field
        for field_name, _ in first_result.items():
            col_name = f"{prefix}.{field_name}"
            values = [getattr(result, field_name) for result in results]
            self.df[col_name] = values
    
    def to_df(self) -> pd.DataFrame:
        """Return the enriched DataFrame."""
        return self.df.copy()


# Simple usage example
if __name__ == "__main__":
    # Example schema
    class Analysis(BaseModel):
        sentiment: str
        summary: str
    
    # Example data
    df = pd.DataFrame({
        "text": ["I love this product!", "This is terrible."]
    })
    
    # Process (requires OPENAI_API_KEY)
    result = (PromptFrame(df)
              .map_prompt("analysis", Analysis)
              .to_df())
    
    print(result)