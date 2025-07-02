"""Utility functions for promptframe."""

from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel


def generate_default_xml_template(df: pd.DataFrame) -> str:
    """
    Generate a default XML template that wraps each column in XML tags.
    
    Args:
        df: DataFrame to generate template for
        
    Returns:
        Jinja template string with XML-wrapped columns
        
    Example:
        For a DataFrame with columns ['name', 'age', 'email'], generates:
        ```
        This is a row of a database, please extract the following information:
        <name>{{ name }}</name>
        <age>{{ age }}</age>
        <email>{{ email }}</email>
        ```
    """
    if df.empty:
        return "This is a row of a database, please extract the information."
    
    xml_tags = []
    for column in df.columns:
        # Sanitize column names for XML (replace spaces/special chars with underscores)
        xml_tag = column.replace(" ", "_").replace("-", "_").replace(".", "_")
        xml_tags.append(f"<{xml_tag}>{{{{ {column} }}}}</{xml_tag}>")
    
    template = (
        "This is a row of a database, please extract the following information:\n"
        + "\n".join(xml_tags)
    )
    return template


def expand_pydantic_to_columns(
    model_instances: List[BaseModel], 
    prefix: str
) -> Dict[str, List[Any]]:
    """
    Expand a list of Pydantic model instances into column data.
    
    Args:
        model_instances: List of Pydantic model instances
        prefix: Column prefix (e.g., "analysis" -> "analysis.summary")
        
    Returns:
        Dictionary mapping column names to lists of values
    """
    if not model_instances:
        return {}
    
    # Get field names from the first instance
    first_instance = model_instances[0]
    first_instance.model_dump()
    
    columns = {}
    
    def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Recursively flatten nested dictionaries."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Handle list of dicts by converting to JSON string
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Process each model instance
    for instance in model_instances:
        flattened = flatten_dict(instance.model_dump())
        
        for key, value in flattened.items():
            col_name = f"{prefix}.{key}" if prefix else key
            if col_name not in columns:
                columns[col_name] = []
            columns[col_name].append(value)
    
    return columns


def merge_columns_to_dataframe(df: pd.DataFrame, new_columns: Dict[str, List[Any]]) -> pd.DataFrame:
    """
    Merge new columns into an existing DataFrame.
    
    Args:
        df: Original DataFrame
        new_columns: Dictionary of column names to values
        
    Returns:
        DataFrame with new columns added
    """
    if not new_columns:
        return df
    
    # Create a new DataFrame with the new columns
    new_df = pd.DataFrame(new_columns, index=df.index)
    
    # Concatenate with original DataFrame
    return pd.concat([df, new_df], axis=1)


def validate_template_kwargs(template_kwargs: Any, row: pd.Series) -> Dict[str, Any]:
    """
    Validate and process template kwargs.
    
    Args:
        template_kwargs: Either a dict, callable returning dict, or None
        row: Current row being processed
        
    Returns:
        Dictionary of template variables
    """
    if template_kwargs is None:
        return {}
    elif callable(template_kwargs):
        return template_kwargs(row)
    elif isinstance(template_kwargs, dict):
        return template_kwargs
    else:
        raise ValueError(f"template_kwargs must be dict, callable, or None, got {type(template_kwargs)}")