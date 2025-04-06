import pandas as pd
import statsmodels.api as sm
import io
from typing import Optional, Any, Dict, List, Union
from pyautocausal.persistence.output_config import OutputType, OutputConfig
from pyautocausal.persistence.parameter_mapper import make_transformable
from .output import StatsmodelsOutputAction
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
import numpy as np
from enum import Enum


class TreatmentType(Enum):
    """Enumeration of treatment types."""
    BINARY = "binary"
    CONTINUOUS = "continuous"
    MULTI_VALUED = "multi_valued"


class ModelSpecification:
    """Base class for model specifications."""
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate that the dataframe contains required columns.
        """
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    @staticmethod
    def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe for modeling (clean data, handle missing values).
        """
        # Make a copy of the dataframe
        df = df.copy()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df


class StandardSpecification(ModelSpecification):
    """Standard regression specification."""
    
    @make_transformable
    @staticmethod
    def action(df: pd.DataFrame, 
               outcome_col: str = 'y', 
               treatment_col: str = 'treat',
               treatment_type: TreatmentType = TreatmentType.BINARY,
               control_cols: Optional[List[str]] = None) -> Dict:
        """
        Create a standard regression specification.
        
        Args:
            df: DataFrame with outcome, treatment, and controls
            outcome_col: Name of outcome column
            treatment_col: Name of treatment column
            treatment_type: Type of treatment (binary, continuous, multi-valued)
            control_cols: List of control variable columns
            
        Returns:
            Dictionary with specification information
        """
        required_columns = [outcome_col, treatment_col]
        ModelSpecification.validate_data(df, required_columns)
        
        # Clean data
        df = ModelSpecification.prepare_data(df)
        
        # If control_cols not specified, use all numeric columns except outcome and treatment
        if control_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            control_cols = [col for col in numeric_cols if col not in [outcome_col, treatment_col]]
        
        # Create specification
        spec = {
            'type': 'standard',
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'treatment_type': treatment_type,
            'control_cols': control_cols,
            'data': df,
            'formula': f"{outcome_col} ~ {treatment_col} + " + " + ".join(control_cols) if control_cols else f"{outcome_col} ~ {treatment_col}"
        }
        
        return spec


class DiDSpecification(ModelSpecification):
    """Difference-in-Differences specification."""
    
    @make_transformable
    @staticmethod
    def action(df: pd.DataFrame, 
               outcome_col: str = 'y', 
               treatment_col: str = 'treat',
               time_col: str = 't',
               unit_col: str = 'id_unit',
               post_col: Optional[str] = None,
               treatment_time_col: Optional[str] = None,
               include_unit_fe: bool = True,
               include_time_fe: bool = True,
               control_cols: Optional[List[str]] = None) -> Dict:
        """
        Create a DiD specification.
        
        Args:
            df: DataFrame with outcome, treatment, time, and unit identifiers
            outcome_col: Name of outcome column
            treatment_col: Name of treatment column
            time_col: Name of time column
            unit_col: Name of unit identifier column
            post_col: Name of post-treatment indicator column
            treatment_time_col: Name of treatment timing column
            include_unit_fe: Whether to include unit fixed effects
            include_time_fe: Whether to include time fixed effects
            control_cols: List of control variable columns
            
        Returns:
            Dictionary with DiD specification information
        """
        required_columns = [outcome_col, treatment_col, time_col, unit_col]
        ModelSpecification.validate_data(df, required_columns)
        
        # Clean data
        df = ModelSpecification.prepare_data(df)
        
        # If control_cols not specified, use all numeric columns except required ones
        if control_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            control_cols = [col for col in numeric_cols if col not in [outcome_col, treatment_col, time_col, unit_col]]
        
        # Create post-treatment indicator if not provided
        if post_col is None:
            if treatment_time_col is not None:
                # Create post indicator based on treatment timing
                df['post'] = (df[time_col] >= df[treatment_time_col]).astype(int)
                post_col = 'post'
            else:
                # Try to infer post periods for treated units
                treat_start = df[df[treatment_col] == 1][time_col].min()
                df['post'] = (df[time_col] >= treat_start).astype(int) if pd.notna(treat_start) else 0
                post_col = 'post'
        
        # Create interaction term
        df['treat_post'] = df[treatment_col] * df[post_col]
        
        # Create specification
        spec = {
            'type': 'did',
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'time_col': time_col,
            'unit_col': unit_col,
            'post_col': post_col,
            'include_unit_fe': include_unit_fe,
            'include_time_fe': include_time_fe,
            'control_cols': control_cols,
            'data': df,
            'interaction_col': 'treat_post'
        }
        
        # Construct formula
        formula_parts = [outcome_col, "~", "treat_post"]
        
        if not include_unit_fe and not include_time_fe:
            # Base model with just the interaction and controls
            formula_parts.extend(["+", treatment_col, "+", post_col])
            if control_cols:
                formula_parts.extend(["+ " + " + ".join(control_cols)])
        else:
            # Model with fixed effects
            if include_unit_fe:
                formula_parts.append("+ C(" + unit_col + ")")
            if include_time_fe:
                formula_parts.append("+ C(" + time_col + ")")
            if control_cols:
                formula_parts.extend(["+ " + " + ".join(control_cols)])
        
        spec['formula'] = " ".join(formula_parts)
        
        return spec


class EventStudySpecification(ModelSpecification):
    """Event Study specification."""
    
    @staticmethod
    def action(df: pd.DataFrame, 
               outcome_col: str = 'y', 
               treatment_col: str = 'treat',
               time_col: str = 't',
               unit_col: str = 'id_unit',
               treatment_time_col: Optional[str] = None,
               relative_time_col: Optional[str] = None,
               pre_periods: int = 3,
               post_periods: int = 3,
               control_cols: Optional[List[str]] = None) -> Dict:
        """
        Create an Event Study specification.
        
        Args:
            df: DataFrame with outcome, treatment, time, and unit identifiers
            outcome_col: Name of outcome column
            treatment_col: Name of treatment column
            time_col: Name of time column
            unit_col: Name of unit identifier column
            treatment_time_col: Name of treatment timing column
            relative_time_col: Name of relative time column
            pre_periods: Number of pre-treatment periods to include
            post_periods: Number of post-treatment periods to include
            control_cols: List of control variable columns
            
        Returns:
            Dictionary with Event Study specification information
        """
        required_columns = [outcome_col, treatment_col, time_col, unit_col]
        ModelSpecification.validate_data(df, required_columns)
        
        # Clean data
        df = ModelSpecification.prepare_data(df)
        
        # Create relative time column if not provided
        if relative_time_col is None:
            # If treatment_time_col is provided, use it to create relative time
            if treatment_time_col is not None:
                df['relative_time'] = df[time_col] - df[treatment_time_col]
                relative_time_col = 'relative_time'
            else:
                # Try to infer treatment timing for each treated unit
                treatment_times = df[df[treatment_col] == 1].groupby(unit_col)[time_col].min()
                
                # Create relative time column
                df['relative_time'] = df.apply(
                    lambda row: row[time_col] - treatment_times.get(row[unit_col], np.nan) 
                    if row[unit_col] in treatment_times.index else np.nan,
                    axis=1
                )
                relative_time_col = 'relative_time'
        
        # Filter to include only the specified pre and post periods
        if pre_periods > 0 or post_periods > 0:
            min_period = -pre_periods if pre_periods > 0 else None
            max_period = post_periods if post_periods > 0 else None
            
            if min_period is not None and max_period is not None:
                df = df[(df[relative_time_col] >= min_period) & (df[relative_time_col] <= max_period)]
            elif min_period is not None:
                df = df[df[relative_time_col] >= min_period]
            elif max_period is not None:
                df = df[df[relative_time_col] <= max_period]
        
        # If control_cols not specified, use all numeric columns except required ones
        if control_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            control_cols = [col for col in numeric_cols if col not in [
                outcome_col, treatment_col, time_col, unit_col, relative_time_col
            ]]
        
        # Create event time dummies
        df = pd.get_dummies(df, columns=[relative_time_col], prefix='event')
        event_cols = [col for col in df.columns if col.startswith('event_')]
        
        # Omit reference period (usually -1) for identification
        omit_period = -1
        omit_col = f'event_{omit_period}'
        if omit_col in event_cols:
            event_cols.remove(omit_col)
        
        # Create specification
        spec = {
            'type': 'event_study',
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'time_col': time_col,
            'unit_col': unit_col,
            'relative_time_col': relative_time_col,
            'event_cols': event_cols,
            'control_cols': control_cols,
            'reference_period': omit_period,
            'data': df
        }
        
        # Construct formula
        formula_parts = [outcome_col, "~"]
        formula_parts.extend(event_cols)
        formula_parts.append("+ C(" + unit_col + ")")
        
        if control_cols:
            formula_parts.extend(["+ " + " + ".join(control_cols)])
        
        spec['formula'] = " + ".join(formula_parts)
        
        return {'data': df, 'specification': spec}
        

class StaggeredDiDSpecification(ModelSpecification):
    """Staggered Difference-in-Differences specification."""
    
    @staticmethod
    def action(df: pd.DataFrame, 
               outcome_col: str = 'y', 
               treatment_col: str = 'treat',
               time_col: str = 't',
               unit_col: str = 'id_unit',
               treatment_time_col: Optional[str] = None,
               control_cols: Optional[List[str]] = None) -> Dict:
        """
        Create a Staggered DiD specification.
        
        Follows the cohort-based approach for staggered treatment adoption.
        
        Args:
            df: DataFrame with outcome, treatment, time, and unit identifiers
            outcome_col: Name of outcome column
            treatment_col: Name of treatment column
            time_col: Name of time column
            unit_col: Name of unit identifier column
            treatment_time_col: Name of treatment timing column
            control_cols: List of control variable columns
            
        Returns:
            Dictionary with Staggered DiD specification information
        """
        required_columns = [outcome_col, treatment_col, time_col, unit_col]
        ModelSpecification.validate_data(df, required_columns)
        
        # Clean data
        df = ModelSpecification.prepare_data(df)
        
        # Determine treatment cohorts
        if treatment_time_col is None:
            # Infer treatment timing from the data
            treatment_times = df[df[treatment_col] == 1].groupby(unit_col)[time_col].min()
            df['treatment_time'] = df[unit_col].map(treatment_times)
            treatment_time_col = 'treatment_time'
        
        # Identify cohorts based on treatment timing
        cohorts = df.dropna(subset=[treatment_time_col])[treatment_time_col].unique()
        cohorts = sorted(cohorts)
        
        # Create cohort indicators
        for cohort in cohorts:
            cohort_col = f'cohort_{cohort}'
            df[cohort_col] = (df[treatment_time_col] == cohort).astype(int)
        
        cohort_cols = [f'cohort_{cohort}' for cohort in cohorts]
        
        # Create post-treatment indicators for each cohort
        for cohort in cohorts:
            post_col = f'post_{cohort}'
            df[post_col] = (df[time_col] >= cohort).astype(int)
        
        post_cols = [f'post_{cohort}' for cohort in cohorts]
        
        # Create cohort-specific treatment effects
        interaction_cols = []
        for cohort in cohorts:
            interaction_col = f'effect_{cohort}'
            df[interaction_col] = df[f'cohort_{cohort}'] * df[f'post_{cohort}']
            interaction_cols.append(interaction_col)
        
        # If control_cols not specified, use all numeric columns except generated ones
        if control_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            excluded_cols = [outcome_col, treatment_col, time_col, unit_col, treatment_time_col] + cohort_cols + post_cols + interaction_cols
            control_cols = [col for col in numeric_cols if col not in excluded_cols]
        
        # Create specification
        spec = {
            'type': 'staggered_did',
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'time_col': time_col,
            'unit_col': unit_col,
            'treatment_time_col': treatment_time_col,
            'cohorts': cohorts,
            'cohort_cols': cohort_cols,
            'post_cols': post_cols,
            'interaction_cols': interaction_cols,
            'control_cols': control_cols,
            'data': df
        }
        
        # Construct formula
        formula_parts = [outcome_col, "~", " + ".join(interaction_cols)]
        
        # Add unit and time fixed effects
        formula_parts.append("+ C(" + unit_col + ")")
        formula_parts.append("+ C(" + time_col + ")")
        
        if control_cols:
            formula_parts.extend(["+ " + " + ".join(control_cols)])
        
        spec['formula'] = " + ".join(formula_parts)
        
        return spec


class ContinuousTreatmentSpecification(ModelSpecification):
    """Continuous Treatment specification."""
    
    @staticmethod
    def action(df: pd.DataFrame, 
               outcome_col: str = 'y', 
               treatment_col: str = 'treat',
               time_col: Optional[str] = None,
               unit_col: Optional[str] = None,
               is_did: bool = False,
               post_col: Optional[str] = None,
               control_cols: Optional[List[str]] = None) -> Dict:
        """
        Create a specification for models with continuous treatment.
        
        Args:
            df: DataFrame with outcome and continuous treatment
            outcome_col: Name of outcome column
            treatment_col: Name of treatment column
            time_col: Name of time column (for panel data)
            unit_col: Name of unit identifier column (for panel data)
            is_did: Whether to use DiD framework with continuous treatment
            post_col: Name of post-treatment indicator column (for DiD)
            control_cols: List of control variable columns
            
        Returns:
            Dictionary with continuous treatment specification information
        """
        required_columns = [outcome_col, treatment_col]
        if is_did:
            required_columns.extend([time_col, unit_col])
            if post_col is not None:
                required_columns.append(post_col)
                
        ModelSpecification.validate_data(df, required_columns)
        
        # Clean data
        df = ModelSpecification.prepare_data(df)
        
        # If control_cols not specified, use all numeric columns except required ones
        if control_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            control_cols = [col for col in numeric_cols if col not in required_columns]
        
        # Create specification
        spec = {
            'type': 'continuous_treatment',
            'outcome_col': outcome_col,
            'treatment_col': treatment_col,
            'control_cols': control_cols,
            'data': df
        }
        
        if is_did:
            spec['type'] = 'continuous_did'
            spec['time_col'] = time_col
            spec['unit_col'] = unit_col
            
            # Create post-treatment indicator if not provided
            if post_col is None:
                # Try to infer post periods
                treat_threshold = df[treatment_col].quantile(0.75)  # Example threshold
                high_treat_units = df[df[treatment_col] > treat_threshold][unit_col].unique()
                treat_start = df[df[unit_col].isin(high_treat_units)][time_col].min()
                
                df['post'] = (df[time_col] >= treat_start).astype(int) if pd.notna(treat_start) else 0
                post_col = 'post'
            
            spec['post_col'] = post_col
            
            # Create interaction term
            df['treat_post'] = df[treatment_col] * df[post_col]
            spec['interaction_col'] = 'treat_post'
            
            # Construct formula for DiD with continuous treatment
            formula_parts = [outcome_col, "~", "treat_post", "+", treatment_col, "+", post_col]
            formula_parts.append("+ C(" + unit_col + ")")
            formula_parts.append("+ C(" + time_col + ")")
            
            if control_cols:
                formula_parts.extend(["+ " + " + ".join(control_cols)])
                
            spec['formula'] = " ".join(formula_parts)
        else:
            # Construct formula for standard regression with continuous treatment
            formula_parts = [outcome_col, "~", treatment_col]
            
            if control_cols:
                formula_parts.extend(["+ " + " + ".join(control_cols)])
                
            spec['formula'] = " ".join(formula_parts)
        
        return spec
        