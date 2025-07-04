"""Base classes for the data cleaning system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

from ..data_validation.base import CleaningHint
from ..data_validation.validator_node import AggregatedValidationResult


@dataclass
class TransformationRecord:
    """Record of a single transformation applied to the data."""
    operation_name: str
    timestamp: datetime
    rows_added: int = 0
    rows_dropped: int = 0
    columns_added: List[str] = field(default_factory=list)
    columns_dropped: List[str] = field(default_factory=list)
    columns_modified: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningMetadata:
    """Metadata about the entire cleaning process."""
    total_operations: int = 0
    total_rows_added: int = 0
    total_rows_dropped: int = 0
    total_columns_added: int = 0
    total_columns_dropped: int = 0
    transformations: List[TransformationRecord] = field(default_factory=list)
    start_shape: Optional[Tuple[int, int]] = None
    end_shape: Optional[Tuple[int, int]] = None
    duration_seconds: Optional[float] = None
    
    def add_transformation(self, record: TransformationRecord):
        """Add a transformation record and update totals."""
        self.transformations.append(record)
        self.total_operations += 1
        self.total_rows_added += record.rows_added
        self.total_rows_dropped += record.rows_dropped
        self.total_columns_added += len(record.columns_added)
        self.total_columns_dropped += len(record.columns_dropped)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "total_operations": self.total_operations,
            "total_rows_added": self.total_rows_added,
            "total_rows_dropped": self.total_rows_dropped,
            "total_columns_added": self.total_columns_added,
            "total_columns_dropped": self.total_columns_dropped,
            "start_shape": self.start_shape,
            "end_shape": self.end_shape,
            "duration_seconds": self.duration_seconds,
            "transformations": [
                {
                    "operation": t.operation_name,
                    "timestamp": t.timestamp.isoformat(),
                    "rows_added": t.rows_added,
                    "rows_dropped": t.rows_dropped,
                    "columns_added": t.columns_added,
                    "columns_dropped": t.columns_dropped,
                    "columns_modified": t.columns_modified,
                    "details": t.details
                }
                for t in self.transformations
            ]
        }


class CleaningOperation(ABC):
    """Abstract base class for cleaning operations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this cleaning operation."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority for ordering operations (higher = executed first)."""
        pass
    
    @abstractmethod
    def can_apply(self, hint: CleaningHint) -> bool:
        """Check if this operation can handle the given cleaning hint."""
        pass
    
    @abstractmethod
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Apply the cleaning operation and return the cleaned DataFrame and transformation record."""
        pass


@dataclass
class CleaningPlan:
    """A plan for cleaning data based on validation results.
    
    This is a callable object that can be applied to a DataFrame.
    """
    operations: List[Tuple[CleaningOperation, CleaningHint]] = field(default_factory=list)
    validation_results: Optional[AggregatedValidationResult] = None
    
    def add_operation(self, operation: CleaningOperation, hint: CleaningHint):
        """Add an operation to the plan."""
        self.operations.append((operation, hint))
    
    def sort_operations(self):
        """Sort operations by priority (higher priority first)."""
        self.operations.sort(key=lambda x: x[0].priority, reverse=True)
    
    def describe(self) -> str:
        """Get a human-readable description of the plan."""
        if not self.operations:
            return "No cleaning operations planned."
        
        lines = ["Cleaning Plan:"]
        lines.append("=" * 50)
        for i, (op, hint) in enumerate(self.operations, 1):
            lines.append(f"{i}. {op.name}")
            if hint.target_columns:
                lines.append(f"   Columns: {', '.join(hint.target_columns)}")
            if hint.parameters:
                lines.append(f"   Parameters: {hint.parameters}")
        return "\n".join(lines)
    
    def __call__(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningMetadata]:
        """Execute the cleaning plan on a DataFrame.
        
        Args:
            df: The DataFrame to clean
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning metadata)
        """
        from time import time
        start_time = time()
        
        metadata = CleaningMetadata(start_shape=df.shape)
        cleaned_df = df.copy()
        
        # Execute operations in order
        for operation, hint in self.operations:
            try:
                cleaned_df, record = operation.apply(cleaned_df, hint)
                metadata.add_transformation(record)
            except Exception as e:
                # Add error record
                error_record = TransformationRecord(
                    operation_name=operation.name,
                    timestamp=datetime.now(),
                    details={"error": str(e), "hint": hint}
                )
                metadata.add_transformation(error_record)
                # Continue with other operations
        
        metadata.end_shape = cleaned_df.shape
        metadata.duration_seconds = time() - start_time
        
        return cleaned_df, metadata 