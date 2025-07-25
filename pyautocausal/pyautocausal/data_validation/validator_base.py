"""DataValidator node for PyAutoCausal.

This module defines objects that aggregate and execute multiple data validation checks on a pandas DataFrame,
returning a combined result. It is intended for use within PyAutoCausal graph pipelines to ensure data quality
and provide detailed validation feedback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union, Any
import pandas as pd

from .base import (
    DataValidationCheck,
    DataValidationConfig,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)


@dataclass
class AggregatedValidationResult:
    """Aggregated result from multiple validation checks.
    
    Attributes:
        passed: Whether all validation checks passed (based on aggregation strategy)
        individual_results: Results from each individual validation check
        summary: Summary of validation results
    """
    passed: bool
    individual_results: List[DataValidationResult]
    summary: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate summary statistics."""
        self.summary = {
            'total_checks': len(self.individual_results),
            'passed_checks': sum(1 for r in self.individual_results if r.passed),
            'failed_checks': sum(1 for r in self.individual_results if not r.passed),
            'total_errors': sum(len(r.get_issues_by_severity(ValidationSeverity.ERROR)) 
                               for r in self.individual_results),
            'total_warnings': sum(len(r.get_issues_by_severity(ValidationSeverity.WARNING)) 
                                 for r in self.individual_results),
            'total_info': sum(len(r.get_issues_by_severity(ValidationSeverity.INFO)) 
                             for r in self.individual_results),
        }
    
    def get_all_issues(self) -> List[ValidationIssue]:
        """Get all issues from all validation checks."""
        issues = []
        for result in self.individual_results:
            issues.extend(result.issues)
        return issues
    
    def get_failed_checks(self) -> List[str]:
        """Get names of all failed validation checks."""
        return [r.check_name for r in self.individual_results if not r.passed]
    
    def to_string(self, verbose: bool = False) -> str:
        """Convert the aggregated result to a human-readable string.
        
        Args:
            verbose: If True, include details of all issues
            
        Returns:
            Formatted string representation of the validation results
        """
        lines = [
            "Data Validation Summary",
            "=" * 50,
            f"Overall Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Checks Run: {self.summary['total_checks']}",
            f"Passed: {self.summary['passed_checks']}",
            f"Failed: {self.summary['failed_checks']}",
            "",
            f"Issues Found:",
            f"  Errors: {self.summary['total_errors']}",
            f"  Warnings: {self.summary['total_warnings']}",
            f"  Info: {self.summary['total_info']}",
        ]
        
        if self.summary['failed_checks'] > 0:
            lines.extend([
                "",
                "Failed Checks:",
                *[f"  - {check}" for check in self.get_failed_checks()]
            ])
        
        if verbose:
            lines.extend(["", "Detailed Issues:", "-" * 50])
            for result in self.individual_results:
                if result.issues:
                    lines.append(f"\n{result.check_name}:")
                    for issue in result.issues:
                        lines.append(f"  [{issue.severity.name.upper()}] {issue.message}")
                        if issue.affected_columns:
                            lines.append(f"    Affected columns: {', '.join(issue.affected_columns)}")
                        if issue.details:
                            lines.append(f"    Details: {issue.details}")
        
        return "\n".join(lines)


@dataclass
class DataValidatorConfig:
    """Configuration for the DataValidatorNode.
    
    This configuration combines:
    1. Node-level settings (aggregation strategy, fail behavior)
    2. Individual check configurations
    """
    # Node-level configuration
    fail_on_error: bool = True  # Whether to fail the node if any ERROR-level issues are found
    fail_on_warning: bool = False  # Whether to fail the node if any WARNING-level issues are found
    aggregation_strategy: str = "all"  # "all" = all checks must pass, "any" = at least one must pass
    
    # Individual check configurations (check_name -> config)
    check_configs: Dict[str, DataValidationConfig] = field(default_factory=dict)
    
    # Which checks to run (if None, run all registered checks)
    enabled_checks: Optional[List[str]] = None


class DataValidator:
    """A node that performs multiple data validation checks on a DataFrame.
    
    This class is designed to be used as the action function for a PyAutoCausal node.
    It aggregates multiple validation checks and returns a combined result.
    """
    
    def __init__(self,
                 checks: List[Union[DataValidationCheck, Type[DataValidationCheck]]],
                 config: Optional[Union[DataValidatorConfig, Dict[str, Any]]] = None):
        """Initialize the validator node with a list of checks.
        
        Args:
            checks: List of validation check instances or classes
            config: Configuration for the validator node
        """
        # Handle different config types
        if isinstance(config, dict):
            self.config = DataValidatorConfig(check_configs=config)
        else:
            self.config = config or DataValidatorConfig()
            
        self.checks: List[DataValidationCheck] = []

        # Initialize checks
        for check in checks:
            if isinstance(check, type):
                # It's a class, instantiate it
                check_name = check().name  # Temporary instance to get name
                check_config = self.config.check_configs.get(check_name)
                self.checks.append(check(config=check_config))
            else:
                # It's already an instance
                # Override config if provided
                if check.name in self.config.check_configs:
                    check.config = self.config.check_configs[check.name]
                self.checks.append(check)
    
    def validate(self, df: pd.DataFrame) -> AggregatedValidationResult:
        """Run all validation checks on the DataFrame.
        
        Args:
            df: The DataFrame to validate
            
        Returns:
            AggregatedValidationResult containing results from all checks
        """
        results = []
        
        # Determine which checks to run
        checks_to_run = self.checks
        if self.config.enabled_checks is not None:
            enabled_names = set(self.config.enabled_checks)
            checks_to_run = [c for c in self.checks if c.name in enabled_names]
        
        # Run each check
        for check in checks_to_run:
            if check.config.enabled:
                result = check.validate(df)
                results.append(result)
        
        # Aggregate results based on strategy
        if self.config.aggregation_strategy == "all":
            # All checks must pass
            passed = all(r.passed for r in results)
        elif self.config.aggregation_strategy == "any":
            # At least one check must pass
            passed = any(r.passed for r in results) if results else False
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.config.aggregation_strategy}")
        
        # Apply fail_on_error and fail_on_warning settings
        if self.config.fail_on_error:
            has_errors = any(r.has_errors for r in results)
            if has_errors:
                passed = False
        
        if self.config.fail_on_warning:
            has_warnings = any(r.has_warnings for r in results)
            if has_warnings:
                passed = False
        
        return AggregatedValidationResult(
            passed=passed,
            individual_results=results
        )
    
    def __call__(self, df: pd.DataFrame) -> AggregatedValidationResult:
        """Make the validator node callable for use as an action function."""
        return self.validate(df) 