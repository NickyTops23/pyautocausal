"""
Conditions Module for Causal Inference Analysis

This module provides a collection of condition checks used in causal inference analysis.
These conditions help validate assumptions and requirements for different causal inference methods.

The module is organized into several classes:
- TimeConditions: Checks related to time series aspects (periods, staggered treatment)
- TreatmentConditions: Checks related to treatment variables (unit counts, balance, treatment type)
- PeriodConditions: Checks for pre/post treatment periods
- CommonChecks: Convenience wrappers for frequently used condition combinations

Treatment Types:
    The module supports both binary (0/1) and continuous treatment variables:
    - Binary treatment: Units are either treated (1) or control (0)
    - Continuous treatment: Treatment intensity varies continuously
        - Uses threshold (default: 0.5) to define treated/control groups
        - Treated: treatment > threshold
        - Control: treatment <= threshold

Example Usage:
    from pyautocausal.orchestration.graph_builder import GraphBuilder
    from pyautocausal.pipelines.library.conditions import CommonChecks
    from pyautocausal.pipelines.library import models
    
    # Create a graph with condition nodes and models
    graph = (GraphBuilder()
                .add_input_node("df")
                # Check if data has multiple periods
                .create_node(
                    name="one_period_check",
                    action_function=lambda x: x,
                    condition=CommonChecks.check_number_of_periods(),
                    predecessors={'df': 'df'}
                )
                # Run OLS if data is balanced
                .create_node(
                    name="run_ols",
                    action_function=models.OLSNode().action,
                    predecessors={'df': 'one_period_check'}
                )
                # Run PSM if data is imbalanced
                .create_node(
                    name="run_psm",
                    action_function=models.PSMNode().action,
                    condition=CommonChecks.check_imbalance(),
                    predecessors={'df': 'one_period_check'}
                ))

Notes:
    - All conditions expect a pandas DataFrame with specific columns:
        - 'y': Outcome variable
        - 't': Time period identifier
        - 'treat': Treatment indicator (binary or continuous)
        - 'id_unit': Unit identifier
    - Most methods will raise ValueError if required columns are missing
    - Use CommonChecks for standard analysis workflows
    - Use specific condition classes for more granular control
    - Conditions are typically used with GraphBuilder to create conditional execution paths
"""

from typing import Callable
import pandas as pd
import numpy as np
from ...orchestration.condition import Condition, create_condition


def index_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index time relative to treatment start (t=0 at treatment start).
    
    Args:
        df: DataFrame with 't', 'treat', and 'id_unit' columns
        
    Returns:
        DataFrame with new 'relative_time' column
    """
    if 'id_unit' not in df.columns:
        raise ValueError("DataFrame must contain 'id_unit' column")
    if 'treat' not in df.columns:
        raise ValueError("DataFrame must contain 'treat' column")
    if 't' not in df.columns:
        return df.assign(relative_time=0)  # Single period case
    
    # Find first treatment time for each unit
    treatment_starts = df[df['treat'] == 1].groupby('id_unit')['t'].min()
    
    # Create relative time column
    df = df.copy()
    df['relative_time'] = df.apply(
        lambda row: row['t'] - treatment_starts[row['id_unit']] 
        if row['id_unit'] in treatment_starts.index 
        else np.nan, 
        axis=1
    )
    
    return df


def is_continuous(series: pd.Series) -> bool:
    """
    Check if a series contains continuous values (not just 0s and 1s).
    
    Args:
        series: pandas Series to check
        
    Returns:
        bool: True if series contains values other than 0 and 1
    """
    unique_values = series.unique()
    return (
        len(unique_values) > 2 or 
        (len(unique_values) == 2 and not all(v in [0, 1] for v in unique_values))
    )


class ConditionFactory:
    """Factory for creating positive and negative conditions."""
    
    @staticmethod
    def create_true_condition(check_func, message):
        """Create a condition that expects the check to return True."""
        return create_condition(check_func, message)
    
    @staticmethod
    def create_false_condition(check_func, message):
        """Create a condition that expects the check to return False."""
        # Create a new function with the same signature as the original function
        def inverted_check(df):
            return not check_func(df)
        
        return create_condition(inverted_check, f"NOT {message}")


class TimeConditions:
    """Class containing conditions related to time series data"""
    
    @staticmethod
    def min_periods(min_periods: int = 3) -> Condition:
        """Condition checking if there are enough time periods"""
        def check_periods(df: pd.DataFrame) -> bool:
            if 't' not in df.columns:
                return False  # Single period case
            return df['t'].nunique() >= min_periods
        
        return create_condition(
            check_periods,
            f"At least {min_periods} time periods required"
        )
    
    @staticmethod
    def is_staggered_treatment() -> Condition:
        """Condition checking if treatment is staggered across time"""
        def check_staggered(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
            if 'id_unit' not in df.columns:
                raise ValueError("DataFrame must contain 'id_unit' column")
            if 't' not in df.columns:
                return False  # Single period case
                
            # Check if treatment starts at different times for different units
            treatment_starts = df.groupby('id_unit')['treat'].apply(
                lambda x: x.eq(1).idxmax() if x.eq(1).any() else None
            )
            return len(treatment_starts.unique()) > 1
        
        return create_condition(
            check_staggered,
            "Treatment must be staggered across time periods"
        )

    @staticmethod
    def multiple_periods(expected=True) -> Condition:
        """Check if data has multiple time periods."""
        def check_periods(df: pd.DataFrame) -> bool:
            if 't' not in df.columns:
                return False  # Single period case
            has_multiple = df['t'].nunique() > 1
            return has_multiple if expected else not has_multiple
        
        message = "Multiple time periods" if expected else "Single time period"
        return create_condition(check_periods, message)


class TreatmentConditions:
    """Class containing conditions related to treatment variables"""
    
    @staticmethod
    def is_continuous_treatment() -> Condition:
        """Check if treatment is continuous rather than binary"""
        def check_continuous(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
            return is_continuous(df['treat'])
        
        return create_condition(
            check_continuous,
            "Checking if treatment is continuous"
        )
    
    @staticmethod
    def min_treated_units(min_units: int = 30, threshold: float = 0.5) -> Condition:
        """
        Condition checking if there are enough treated units.
        For continuous treatment, units are considered treated if treatment > threshold.
        
        Args:
            min_units: Minimum number of treated units required
            threshold: Treatment threshold for continuous treatment (default: 0.5)
        """
        def check_treated_units(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
            if is_continuous(df['treat']):
                return (df['treat'] > threshold).sum() >= min_units
            return (df['treat'] == 1).sum() >= min_units
        
        return create_condition(
            check_treated_units,
            f"At least {min_units} treated units required"
        )
    
    @staticmethod
    def min_control_units(min_units: int = 30, threshold: float = 0.5) -> Condition:
        """
        Condition checking if there are enough control units.
        For continuous treatment, units are considered control if treatment <= threshold.
        
        Args:
            min_units: Minimum number of control units required
            threshold: Treatment threshold for continuous treatment (default: 0.5)
        """
        def check_control_units(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
            if is_continuous(df['treat']):
                return (df['treat'] <= threshold).sum() >= min_units
            return (df['treat'] == 0).sum() >= min_units
        
        return create_condition(
            check_control_units,
            f"At least {min_units} control units required"
        )

    @staticmethod
    def check_imbalance(threshold: float = 0.5) -> Condition:
        """
        Check if there is covariate imbalance between treatment and control groups.
        For continuous treatment, uses threshold to define groups.
        
        Args:
            threshold: Treatment threshold for continuous treatment (default: 0.5)
        """
        def check_balance(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
                
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            covariates = [col for col in numeric_cols if col not in ['treat', 't', 'id_unit']]
            
            if not covariates:
                return False
            
            # Define treatment groups based on continuous or binary treatment
            if is_continuous(df['treat']):
                treated_mask = df['treat'] > threshold
            else:
                treated_mask = df['treat'] == 1
                
            for col in covariates:
                treated_mean = df[treated_mask][col].mean()
                control_mean = df[~treated_mask][col].mean()
                treated_var = df[treated_mask][col].var()
                control_var = df[~treated_mask][col].var()
                
                std_diff = abs(treated_mean - control_mean) / np.sqrt((treated_var + control_var) / 2)
                
                if std_diff > 0.25:
                    return True
                    
            return False
        
        return create_condition(
            check_balance,
            "Checking for covariate imbalance between treatment and control groups"
        )

    @staticmethod
    def multiple_treated_units(expected=True) -> Condition:
        """Check if data has multiple treated units."""
        def check_treated(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
            has_multiple = (df['treat'] == 1).sum() > 1
            return has_multiple if expected else not has_multiple
        
        message = "Multiple treated units" if expected else "Single treated unit"
        return create_condition(check_treated, message)
    
    @staticmethod
    def imbalanced_covariates(expected=True, threshold: float = 0.5) -> Condition:
        """Check if there is covariate imbalance between treatment and control groups."""
        def check_balance(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
                
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            covariates = [col for col in numeric_cols if col not in ['treat', 't', 'id_unit']]
            
            if not covariates:
                return False
            
            # Define treatment groups based on continuous or binary treatment
            if is_continuous(df['treat']):
                treated_mask = df['treat'] > threshold
            else:
                treated_mask = df['treat'] == 1
                
            for col in covariates:
                treated_mean = df[treated_mask][col].mean()
                control_mean = df[~treated_mask][col].mean()
                treated_var = df[treated_mask][col].var()
                control_var = df[~treated_mask][col].var()
                
                std_diff = abs(treated_mean - control_mean) / np.sqrt((treated_var + control_var) / 2)
                
                if std_diff > 0.25:
                    return True
                    
            return False
        
        message = "Imbalanced covariates" if expected else "Balanced covariates"
        return create_condition(check_balance, message)


class PeriodConditions:
    """Class containing conditions related to pre/post treatment periods"""
    
    @staticmethod
    def min_pre_periods(min_periods: int = 3) -> Condition:
        """Check if there are enough pre-treatment periods"""
        def check_pre_periods(df: pd.DataFrame) -> bool:
            df = index_relative_time(df)
            return df[df['relative_time'] < 0]['relative_time'].nunique() >= min_periods
        
        return create_condition(
            check_pre_periods,
            f"At least {min_periods} pre-treatment periods required"
        )
    
    @staticmethod
    def min_post_periods(min_periods: int = 3) -> Condition:
        """Check if there are enough post-treatment periods"""
        def check_post_periods(df: pd.DataFrame) -> bool:
            df = index_relative_time(df)
            return df[df['relative_time'] >= 0]['relative_time'].nunique() >= min_periods
        
        return create_condition(
            check_post_periods,
            f"At least {min_periods} post-treatment periods required"
        )

    @staticmethod
    def sufficient_pre_periods(expected=True, min_periods=2) -> Condition:
        """Check if there are enough pre-treatment periods."""
        def check_pre_periods(df: pd.DataFrame) -> bool:
            df = index_relative_time(df)
            has_sufficient = df[df['relative_time'] < 0]['relative_time'].nunique() >= min_periods
            return has_sufficient if expected else not has_sufficient
        
        message = f"At least {min_periods} pre-treatment periods"
        return create_condition(check_pre_periods, message)
    
    @staticmethod
    def sufficient_post_periods(expected=True, min_periods=2) -> Condition:
        """Check if there are enough post-treatment periods."""
        def check_post_periods(df: pd.DataFrame) -> bool:
            df = index_relative_time(df)
            has_sufficient = df[df['relative_time'] >= 0]['relative_time'].nunique() >= min_periods
            return has_sufficient if expected else not has_sufficient
        
        message = f"At least {min_periods} post-treatment periods"
        return create_condition(check_post_periods, message)

    @staticmethod
    def staggered_treatment(expected=True) -> Condition:
        """Check if treatment timing is staggered across units."""
        def check_staggered(df: pd.DataFrame) -> bool:
            if 'treat' not in df.columns:
                raise ValueError("DataFrame must contain 'treat' column")
            if 'id_unit' not in df.columns:
                raise ValueError("DataFrame must contain 'id_unit' column")
            if 't' not in df.columns:
                return False  # Single period case
                
            # Check if treatment starts at different times for different units
            treatment_starts = df.groupby('id_unit')['treat'].apply(
                lambda x: x.eq(1).idxmax() if x.eq(1).any() else None
            )
            is_staggered = len(treatment_starts.unique()) > 1
            return is_staggered if expected else not is_staggered
        
        message = "Staggered treatment" if expected else "Non-staggered treatment"
        return create_condition(check_staggered, message)


# Convenience class that combines commonly used checks
class CommonChecks:
    """Common condition checks used in causal inference"""
    
    @staticmethod
    def check_required_columns() -> Condition:
        """Check if required columns (treat, id_unit) exist in the DataFrame"""
        def check_columns(df: pd.DataFrame) -> bool:
            required_columns = ['y', 'treat', 'id_unit']
            return all(col in df.columns for col in required_columns)
        
        return create_condition(
            check_columns,
            "Checking for required columns: treat, id_unit"
        )

    @staticmethod
    def check_number_of_periods() -> Condition:
        return TimeConditions.multiple_periods(expected=True)

    @staticmethod
    def check_number_of_treated_units() -> Condition:
        return TreatmentConditions.multiple_treated_units(expected=True)

    @staticmethod
    def check_number_of_pre_periods() -> Condition:
        return PeriodConditions.sufficient_pre_periods(expected=True, min_periods=2)

    @staticmethod
    def check_number_of_post_periods() -> Condition:
        return PeriodConditions.sufficient_post_periods(expected=True, min_periods=2)

    @staticmethod
    def check_staggered() -> Condition:
        return TimeConditions.staggered_treatment(expected=True)

    @staticmethod
    def check_imbalance() -> Condition:
        return TreatmentConditions.imbalanced_covariates(expected=True)

    @staticmethod
    def check_continuous_treatment() -> Condition:
        return TreatmentConditions.is_continuous_treatment()


def get_condition_checks():
    """Returns a dictionary of all available condition checks"""
    return {
        'check_required_columns': CommonChecks.check_required_columns,
        'check_number_of_periods': CommonChecks.check_number_of_periods,
        'check_number_of_treated_units': CommonChecks.check_number_of_treated_units,
        'check_number_of_pre_periods': CommonChecks.check_number_of_pre_periods,
        'check_number_of_post_periods': CommonChecks.check_number_of_post_periods,
        'check_staggered': CommonChecks.check_staggered,
        'check_imbalance': CommonChecks.check_imbalance,
        'check_continuous_treatment': CommonChecks.check_continuous_treatment
    }

