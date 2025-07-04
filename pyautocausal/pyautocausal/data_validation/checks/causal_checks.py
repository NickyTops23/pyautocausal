"""Causal inference specific validation checks for pandas DataFrames."""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union
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
class BinaryTreatmentConfig(DataValidationConfig):
    """Configuration for BinaryTreatmentCheck."""
    treatment_column: str = "treatment"
    allow_missing: bool = False
    valid_values: Set[Union[int, float]] = field(default_factory=lambda: {0, 1})


class BinaryTreatmentCheck(DataValidationCheck[BinaryTreatmentConfig]):
    """Check that treatment variable is binary (0/1)."""
    
    @property
    def name(self) -> str:
        return "binary_treatment"
    
    @classmethod
    def get_default_config(cls) -> BinaryTreatmentConfig:
        return BinaryTreatmentConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if treatment column exists
        if self.config.treatment_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Treatment column '{self.config.treatment_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        treatment = df[self.config.treatment_column]
        
        # Check for missing values
        missing_count = treatment.isna().sum()
        if missing_count > 0 and not self.config.allow_missing:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment column has {missing_count} missing values",
                affected_columns=[self.config.treatment_column],
                details={"missing_count": int(missing_count)}
            ))
        
        # Check values are in valid set
        unique_values = set(treatment.dropna().unique())
        invalid_values = unique_values - self.config.valid_values
        
        if invalid_values:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment column contains invalid values: {invalid_values}. Expected: {self.config.valid_values}",
                affected_columns=[self.config.treatment_column],
                details={
                    "invalid_values": list(invalid_values),
                    "valid_values": list(self.config.valid_values),
                    "unique_values": list(unique_values)
                }
            ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        
        # Calculate treatment statistics
        value_counts = treatment.value_counts().to_dict()
        metadata = {
            "treatment_column": self.config.treatment_column,
            "unique_values": list(unique_values),
            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
            "missing_count": int(missing_count),
            "treatment_fraction": float(treatment.sum() / len(treatment.dropna())) if len(treatment.dropna()) > 0 else None
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TreatmentVariationConfig(DataValidationConfig):
    """Configuration for TreatmentVariationCheck."""
    treatment_column: str = "treatment"
    min_treated_fraction: float = 0.05  # Minimum fraction of treated units
    max_treated_fraction: float = 0.95  # Maximum fraction of treated units
    min_treated_count: int = 10  # Minimum number of treated units
    min_control_count: int = 10  # Minimum number of control units


class TreatmentVariationCheck(DataValidationCheck[TreatmentVariationConfig]):
    """Check that there is sufficient variation in treatment assignment."""
    
    @property
    def name(self) -> str:
        return "treatment_variation"
    
    @classmethod
    def get_default_config(cls) -> TreatmentVariationConfig:
        return TreatmentVariationConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if treatment column exists
        if self.config.treatment_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Treatment column '{self.config.treatment_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        treatment = df[self.config.treatment_column].dropna()
        
        # Count treated and control units
        treated_count = (treatment == 1).sum()
        control_count = (treatment == 0).sum()
        total_count = len(treatment)
        
        if total_count == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No non-missing treatment values found",
                affected_columns=[self.config.treatment_column]
            ))
            return self._create_result(False, issues)
        
        treated_fraction = treated_count / total_count
        
        # Check minimum counts
        if treated_count < self.config.min_treated_count:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Only {treated_count} treated units, minimum required is {self.config.min_treated_count}",
                affected_columns=[self.config.treatment_column],
                details={"treated_count": int(treated_count), "required": self.config.min_treated_count}
            ))
        
        if control_count < self.config.min_control_count:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Only {control_count} control units, minimum required is {self.config.min_control_count}",
                affected_columns=[self.config.treatment_column],
                details={"control_count": int(control_count), "required": self.config.min_control_count}
            ))
        
        # Check treatment fraction
        if treated_fraction < self.config.min_treated_fraction:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment fraction {treated_fraction:.1%} is below minimum {self.config.min_treated_fraction:.1%}",
                affected_columns=[self.config.treatment_column],
                details={
                    "treated_fraction": float(treated_fraction),
                    "min_fraction": self.config.min_treated_fraction
                }
            ))
        
        if treated_fraction > self.config.max_treated_fraction:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment fraction {treated_fraction:.1%} is above maximum {self.config.max_treated_fraction:.1%}",
                affected_columns=[self.config.treatment_column],
                details={
                    "treated_fraction": float(treated_fraction),
                    "max_fraction": self.config.max_treated_fraction
                }
            ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "treated_count": int(treated_count),
            "control_count": int(control_count),
            "total_count": int(total_count),
            "treated_fraction": float(treated_fraction)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class PanelStructureConfig(DataValidationConfig):
    """Configuration for PanelStructureCheck."""
    unit_column: str = "unit_id"
    time_column: str = "time"
    require_balanced: bool = True  # Whether to require balanced panel


class PanelStructureCheck(DataValidationCheck[PanelStructureConfig]):
    """Check panel data structure for causal analysis."""
    
    @property
    def name(self) -> str:
        return "panel_structure"
    
    @classmethod
    def get_default_config(cls) -> PanelStructureConfig:
        return PanelStructureConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check required columns exist
        for col, col_name in [(self.config.unit_column, "unit"), (self.config.time_column, "time")]:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"{col_name.capitalize()} column '{col}' not found",
                    details={"available_columns": list(df.columns)}
                ))
        
        if issues:
            return self._create_result(False, issues)
        
        # Get unique units and time periods
        units = df[self.config.unit_column].unique()
        times = df[self.config.time_column].unique()
        n_units = len(units)
        n_times = len(times)
        
        # Check for duplicate unit-time observations
        duplicates = df.duplicated(subset=[self.config.unit_column, self.config.time_column])
        if duplicates.any():
            dup_count = duplicates.sum()
            sample_dups = df[duplicates].head(5)
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Found {dup_count} duplicate unit-time observations",
                affected_rows=sample_dups.index.tolist(),
                details={
                    "duplicate_count": int(dup_count),
                    "sample_duplicates": sample_dups[[self.config.unit_column, self.config.time_column]].to_dict('records')
                }
            ))
        
        # Check if panel is balanced
        expected_obs = n_units * n_times
        actual_obs = len(df.drop_duplicates(subset=[self.config.unit_column, self.config.time_column]))
        is_balanced = expected_obs == actual_obs
        
        if self.config.require_balanced and not is_balanced:
            # Find missing observations
            all_combinations = pd.MultiIndex.from_product([units, times], 
                                                        names=[self.config.unit_column, self.config.time_column])
            existing_combinations = pd.MultiIndex.from_frame(
                df[[self.config.unit_column, self.config.time_column]].drop_duplicates()
            )
            missing_combinations = all_combinations.difference(existing_combinations)
            
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Panel is unbalanced: {len(missing_combinations)} unit-time combinations missing",
                details={
                    "expected_observations": expected_obs,
                    "actual_observations": actual_obs,
                    "missing_count": len(missing_combinations),
                    "sample_missing": [f"{u}-{t}" for u, t in missing_combinations[:5]]
                }
            ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "n_units": n_units,
            "n_times": n_times,
            "n_observations": len(df),
            "is_balanced": is_balanced,
            "balance_ratio": actual_obs / expected_obs if expected_obs > 0 else 0
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TimeColumnConfig(DataValidationConfig):
    """Configuration for TimeColumnCheck."""
    time_column: str = "time"
    require_sequential: bool = True  # Whether time periods should be sequential
    require_numeric: bool = False  # Whether time should be numeric
    date_format: Optional[str] = None  # Expected date format if time is string


class TimeColumnCheck(DataValidationCheck[TimeColumnConfig]):
    """Check time column properties for causal analysis."""
    
    @property
    def name(self) -> str:
        return "time_column"
    
    @classmethod
    def get_default_config(cls) -> TimeColumnConfig:
        return TimeColumnConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if time column exists
        if self.config.time_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Time column '{self.config.time_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        time_col = df[self.config.time_column]
        
        # Check for missing values
        if time_col.isna().any():
            missing_count = time_col.isna().sum()
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Time column has {missing_count} missing values",
                affected_columns=[self.config.time_column],
                details={"missing_count": int(missing_count)}
            ))
        
        # Get unique time periods
        unique_times = sorted(time_col.dropna().unique())
        n_periods = len(unique_times)
        
        # Check data type
        is_numeric = pd.api.types.is_numeric_dtype(time_col)
        is_datetime = pd.api.types.is_datetime64_any_dtype(time_col)
        
        if self.config.require_numeric and not is_numeric:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Time column is not numeric (type: {time_col.dtype})",
                affected_columns=[self.config.time_column],
                details={"actual_type": str(time_col.dtype)}
            ))
        
        # Check sequential periods if required
        if self.config.require_sequential and n_periods > 1:
            if is_numeric:
                # Check for gaps in numeric sequence
                diffs = np.diff(unique_times)
                if not np.all(diffs == diffs[0]):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Time periods are not evenly spaced",
                        affected_columns=[self.config.time_column],
                        details={
                            "period_gaps": [f"{unique_times[i]}-{unique_times[i+1]}: {diffs[i]}" 
                                          for i in range(min(5, len(diffs)))]
                        }
                    ))
            elif is_datetime:
                # Check for regular intervals in datetime
                time_diffs = pd.Series(unique_times).diff()[1:]
                if not time_diffs.nunique() == 1:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Time periods have irregular intervals",
                        affected_columns=[self.config.time_column],
                        details={"unique_intervals": time_diffs.value_counts().to_dict()}
                    ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "n_periods": n_periods,
            "time_range": [str(unique_times[0]), str(unique_times[-1])] if n_periods > 0 else None,
            "is_numeric": is_numeric,
            "is_datetime": is_datetime,
            "unique_periods": [str(t) for t in unique_times[:10]]  # First 10 periods
        }
        
        return self._create_result(passed, issues, metadata) 