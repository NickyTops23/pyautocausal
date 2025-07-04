"""Basic data validation checks for pandas DataFrames."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

from ..base import (
    DataValidationCheck,
    DataValidationConfig,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)


@dataclass
class NonEmptyDataConfig(DataValidationConfig):
    """Configuration for NonEmptyDataCheck."""
    min_rows: int = 1
    min_columns: int = 1


class NonEmptyDataCheck(DataValidationCheck[NonEmptyDataConfig]):
    """Check that the DataFrame is not empty."""
    
    @property
    def name(self) -> str:
        return "non_empty_data"
    
    @classmethod
    def get_default_config(cls) -> NonEmptyDataConfig:
        return NonEmptyDataConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check rows
        if len(df) < self.config.min_rows:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"DataFrame has {len(df)} rows, but minimum required is {self.config.min_rows}",
                details={"actual_rows": len(df), "required_rows": self.config.min_rows}
            ))
        
        # Check columns
        if len(df.columns) < self.config.min_columns:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"DataFrame has {len(df.columns)} columns, but minimum required is {self.config.min_columns}",
                details={"actual_columns": len(df.columns), "required_columns": self.config.min_columns}
            ))
        
        passed = len(issues) == 0
        metadata = {
            "n_rows": len(df),
            "n_columns": len(df.columns)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class RequiredColumnsConfig(DataValidationConfig):
    """Configuration for RequiredColumnsCheck."""
    required_columns: List[str] = None  # Will be set in __post_init__
    case_sensitive: bool = True
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = []


class RequiredColumnsCheck(DataValidationCheck[RequiredColumnsConfig]):
    """Check that required columns are present in the DataFrame."""
    
    @property
    def name(self) -> str:
        return "required_columns"
    
    @classmethod
    def get_default_config(cls) -> RequiredColumnsConfig:
        return RequiredColumnsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        if not self.config.required_columns:
            # No required columns specified, pass by default
            return self._create_result(True, [], {"message": "No required columns specified"})
        
        # Get DataFrame columns
        df_columns = set(df.columns)
        if not self.config.case_sensitive:
            df_columns = {col.lower() for col in df_columns}
        
        # Check each required column
        missing_columns = []
        for required_col in self.config.required_columns:
            check_col = required_col if self.config.case_sensitive else required_col.lower()
            if check_col not in df_columns:
                missing_columns.append(required_col)
        
        if missing_columns:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Missing required columns: {', '.join(missing_columns)}",
                details={"missing_columns": missing_columns}
            ))
        
        passed = len(issues) == 0
        metadata = {
            "required_columns": self.config.required_columns,
            "missing_columns": missing_columns,
            "found_columns": list(df.columns)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class ColumnTypesConfig(DataValidationConfig):
    """Configuration for ColumnTypesCheck."""
    expected_types: Dict[str, type] = None  # Column name -> expected type
    categorical_threshold: int = 10  # Max unique values to consider as categorical
    infer_categorical: bool = True  # Whether to infer categorical columns
    
    def __post_init__(self):
        if self.expected_types is None:
            self.expected_types = {}


class ColumnTypesCheck(DataValidationCheck[ColumnTypesConfig]):
    """Check that columns have expected data types."""
    
    @property
    def name(self) -> str:
        return "column_types"
    
    @classmethod
    def get_default_config(cls) -> ColumnTypesConfig:
        return ColumnTypesConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        actual_types = {}
        inferred_categorical = []
        
        # Check specified column types
        for col, expected_type in self.config.expected_types.items():
            if col not in df.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{col}' not found for type checking",
                    affected_columns=[col]
                ))
                continue
            
            actual_dtype = df[col].dtype
            actual_types[col] = str(actual_dtype)
            
            # Type checking logic
            type_matches = False
            if expected_type == str:
                type_matches = pd.api.types.is_string_dtype(df[col])
            elif expected_type == int:
                type_matches = pd.api.types.is_integer_dtype(df[col])
            elif expected_type == float:
                type_matches = pd.api.types.is_float_dtype(df[col])
            elif expected_type == bool:
                type_matches = pd.api.types.is_bool_dtype(df[col])
            elif expected_type == 'datetime':
                type_matches = pd.api.types.is_datetime64_any_dtype(df[col])
            elif expected_type == 'categorical':
                type_matches = pd.api.types.is_categorical_dtype(df[col]) or \
                              (self.config.infer_categorical and df[col].nunique() <= self.config.categorical_threshold)
            
            if not type_matches:
                issues.append(ValidationIssue(
                    severity=self.config.severity_on_fail,
                    message=f"Column '{col}' has type {actual_dtype}, expected {expected_type}",
                    affected_columns=[col],
                    details={"actual_type": str(actual_dtype), "expected_type": str(expected_type)}
                ))
        
        # Infer categorical columns if enabled
        if self.config.infer_categorical:
            for col in df.columns:
                if col not in self.config.expected_types:
                    if df[col].nunique() <= self.config.categorical_threshold:
                        inferred_categorical.append(col)
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Column '{col}' appears to be categorical ({df[col].nunique()} unique values)",
                            affected_columns=[col],
                            details={"unique_values": df[col].nunique()}
                        ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "actual_types": actual_types,
            "inferred_categorical": inferred_categorical
        }
        
        # Generate cleaning hints
        cleaning_hints = []
        if inferred_categorical:
            from ..base import CleaningHint
            cleaning_hints.append(CleaningHint(
                operation_type="convert_to_categorical",
                target_columns=inferred_categorical,
                priority=90,
                metadata={
                    "categorical_threshold": self.config.categorical_threshold,
                    "unique_counts": {col: df[col].nunique() for col in inferred_categorical}
                }
            ))
        
        return self._create_result(passed, issues, metadata, cleaning_hints)


@dataclass
class NoDuplicateColumnsConfig(DataValidationConfig):
    """Configuration for NoDuplicateColumnsCheck."""
    case_sensitive: bool = True


class NoDuplicateColumnsCheck(DataValidationCheck[NoDuplicateColumnsConfig]):
    """Check that there are no duplicate column names."""
    
    @property
    def name(self) -> str:
        return "no_duplicate_columns"
    
    @classmethod
    def get_default_config(cls) -> NoDuplicateColumnsConfig:
        return NoDuplicateColumnsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check for duplicate column names
        columns = df.columns.tolist()
        if not self.config.case_sensitive:
            columns = [col.lower() for col in columns]
        
        seen = set()
        duplicates = set()
        for col in columns:
            if col in seen:
                duplicates.add(col)
            seen.add(col)
        
        if duplicates:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Duplicate column names found: {', '.join(duplicates)}",
                details={"duplicate_columns": list(duplicates)}
            ))
        
        passed = len(issues) == 0
        metadata = {
            "n_columns": len(df.columns),
            "n_unique_columns": len(set(columns)),
            "duplicates": list(duplicates)
        }
        
        return self._create_result(passed, issues, metadata) 