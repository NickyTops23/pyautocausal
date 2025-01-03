from dataclasses import dataclass
from typing import Optional
from .output_types import OutputType

@dataclass
class OutputConfig:
    """Configuration for node output persistence"""
    save_output: bool = False
    output_filename: Optional[str] = None  # If None and save_output is True, use node name
    output_type: Optional[OutputType] = None  # This cannot