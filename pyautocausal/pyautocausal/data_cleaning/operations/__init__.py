"""Built-in cleaning operations."""

from .categorical_operations import ConvertToCategoricalOperation, EncodeMissingAsCategoryOperation
from .missing_data_operations import DropMissingRowsOperation, FillMissingWithValueOperation
from .duplicate_operations import DropDuplicateRowsOperation

# List of all available operations
ALL_OPERATIONS = [
    ConvertToCategoricalOperation(),
    EncodeMissingAsCategoryOperation(),
    DropMissingRowsOperation(),
    FillMissingWithValueOperation(),
    DropDuplicateRowsOperation(),
]


def get_all_operations():
    """Get instances of all available cleaning operations."""
    return ALL_OPERATIONS.copy()


__all__ = [
    'ConvertToCategoricalOperation',
    'EncodeMissingAsCategoryOperation',
    'DropMissingRowsOperation',
    'FillMissingWithValueOperation',
    'DropDuplicateRowsOperation',
    'get_all_operations',
] 