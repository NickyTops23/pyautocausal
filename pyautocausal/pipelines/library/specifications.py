import pandas as pd
import statsmodels.api as sm
import io
from typing import Optional, Any, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from pyautocausal.persistence.output_config import OutputType, OutputConfig
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.output import write_statsmodels_summary
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
import numpy as np
import patsy
from enum import Enum

@dataclass
class BaseSpec:
    """Base specification with common fields."""
    data: pd.DataFrame
    formula: str


@dataclass
class CrossSectionalSpec(BaseSpec):
    """Univariate specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]

@dataclass
class DiDSpec(BaseSpec):
    """Difference-in-Differences specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    post_col: str
    include_unit_fe: bool
    include_time_fe: bool
    model: Optional[Any] = None

@dataclass
class EventStudySpec(BaseSpec):
    """Event Study specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    relative_time_col: str
    event_cols: List[str]
    reference_period: int
    model: Optional[Any] = None


@dataclass
class StaggeredDiDSpec(BaseSpec):
    """Staggered Difference-in-Differences specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    treatment_time_col: str
    cohorts: List[Any]
    cohort_cols: List[str]
    post_cols: List[str]
    interaction_cols: List[str]
    model: Optional[Any] = None


def validate_and_prepare_data(
    data: pd.DataFrame,
    outcome_col: str,
    treatment_cols: List[str],
    required_columns: List[str] = None,
    control_cols: Optional[List[str]] = None,
    excluded_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and prepare dataframe for specification creation.
    
    Args:
        data: DataFrame to validate and prepare
        outcome_col: Name of outcome column
        treatment_cols: Names of treatment columns
        required_columns: List of additional required columns (beyond outcome and treatment)
        control_cols: Explicitly provided control columns (if None, will be auto-determined)
        excluded_cols: Columns to exclude when auto-determining control columns
        
    Returns:
        Tuple of (cleaned dataframe, control columns list)
    """
    # Handle control_cols=None appropriately
    if control_cols is None:
        control_cols = []  # Default empty list instead of raising error
    
    # Build complete list of required columns
    all_required = [outcome_col] + treatment_cols
    if required_columns:
        all_required.extend(required_columns)
    
    # Check if required columns exist
    missing_columns = [col for col in all_required if col not in data.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing the following required columns: {missing_columns}. All required columns are: {all_required}")
    
    # Clean data (make a copy to avoid modifying the original)
    cleaned_data = data.copy().dropna()
    
    # Determine control columns if not provided
    if not control_cols:
        # Get numeric columns as potential controls
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        # Determine columns to exclude
        to_exclude = [outcome_col] + treatment_cols
        if excluded_cols:
            to_exclude.extend(excluded_cols)
            
        # Filter numeric columns to only include those not in excluded list
        control_cols = [col for col in numeric_cols if col not in to_exclude]
        
    return cleaned_data, control_cols


@make_transformable
def create_cross_sectional_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    control_cols: Optional[List[str]] = None
) -> BaseSpec:
    """
    Create a standard regression specification.
    
    Args:
        data: DataFrame with outcome, treatment, and controls
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names (only first is used for now)
        control_cols: List of control variable columns
        
    Returns:
        BaseSpec object with specification information
    """
    # TODO:Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols
    )
    
    # Standard-specific validation: check if treatment column is binary
    n_unique_treatment = data[treatment_col].nunique()
    if n_unique_treatment > 2:
        raise ValueError(f"Treatment column {treatment_col} must be binary")
    elif n_unique_treatment < 2:
        raise ValueError(f"Treatment column {treatment_col} must have at least two unique values")  
    
    # Make treatment column binary (0/1)
    data[treatment_col] = np.where(data[treatment_col] == data[treatment_col].unique()[0], 0, 1)
    
    # Create formula
    formula = (f"{outcome_col} ~ {treatment_col} + " + " + ".join(control_cols) 
              if control_cols else f"{outcome_col} ~ {treatment_col}")
    
    # return CrossSectionalSpec with treatment_cols
    return CrossSectionalSpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,  # Use the list
        control_cols=control_cols,
        data=data,
        formula=formula
    )


@make_transformable
def create_did_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    post_col: Optional[str] = None,
    treatment_time_col: Optional[str] = None,
    include_unit_fe: bool = True,
    include_time_fe: bool = True,
    control_cols: Optional[List[str]] = None
) -> DiDSpec:
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
        DiDSpec object with DiD specification information
    """
    # TODO: Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols,
        excluded_cols=[time_col, unit_col]
    )
    
    # Create post-treatment indicator if not provided
    if post_col is None:
        if treatment_time_col is not None:
            # Create post indicator based on treatment timing
            data['post'] = (data[time_col] >= data[treatment_time_col]).astype(int)
            post_col = 'post'
        else:
            # Try to infer post periods for treated units
            treat_start = data[data[treatment_col] == 1][time_col].min()
            data['post'] = (data[time_col] >= treat_start).astype(int) if pd.notna(treat_start) else 0
            post_col = 'post'
    
    # Create interaction term
    data['treat_post'] = data[treatment_col] * data[post_col]
    
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
    
    formula = " ".join(formula_parts)
    
    # Create and return specification
    return DiDSpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        time_col=time_col,
        unit_col=unit_col,
        post_col=post_col,
        include_unit_fe=include_unit_fe,
        include_time_fe=include_time_fe,
        control_cols=control_cols,
        data=data,
        formula=formula
    )


@make_transformable
def create_event_study_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    treatment_time_col: Optional[str] = None,
    relative_time_col: Optional[str] = None,
    pre_periods: int = 3,
    post_periods: int = 3,
    control_cols: Optional[List[str]] = None
) -> EventStudySpec:
    """
    Create an Event Study specification.
    
    Args:
        data: DataFrame with outcome, treatment, time, and unit identifiers
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names
        time_col: Name of time column
        unit_col: Name of unit identifier column
        treatment_time_col: Name of treatment timing column
        relative_time_col: Name of relative time column
        pre_periods: Number of pre-treatment periods to include
        post_periods: Number of post-treatment periods to include
        control_cols: List of control variable columns
        
    Returns:
        EventStudySpec object with event study specification information
    """
    # Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    # Ensure 'post' is excluded if it exists to avoid issues with event dummies + time FEs
    current_excluded_cols = [time_col, unit_col]
    if 'post' in data.columns:
        current_excluded_cols.append('post')

    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols, # Pass user-provided or None
        excluded_cols=current_excluded_cols
    )
    
    # Create relative time column if not provided
    if relative_time_col is None:
        # If treatment_time_col is provided, use it to create relative time
        if treatment_time_col is not None:
            data['relative_time'] = data[time_col] - data[treatment_time_col]
            relative_time_col = 'relative_time'
        else:
            # Try to infer treatment timing for each treated unit
            treatment_times = data[data[treatment_col] == 1].groupby(unit_col)[time_col].min()
            
            # Create relative time column
            data['relative_time'] = data.apply(
                lambda row: row[time_col] - treatment_times.get(row[unit_col], np.nan) 
                if row[unit_col] in treatment_times.index else np.nan,
                axis=1
            )
            relative_time_col = 'relative_time'
    
    # Filter to include only the specified pre and post periods
    if pre_periods > 0 or post_periods > 0:
        min_period = -pre_periods if pre_periods > 0 else None
        max_period = post_periods if post_periods > 0 else None
        
        # Keep rows that either match the filter or have NaN in relative_time_col
        if min_period is not None and max_period is not None:
            data = data[(data[relative_time_col].isna()) | 
                       ((data[relative_time_col] >= min_period) & (data[relative_time_col] <= max_period))]
        elif min_period is not None:
            data = data[(data[relative_time_col].isna()) | (data[relative_time_col] >= min_period)]
        elif max_period is not None:
            data = data[(data[relative_time_col].isna()) | (data[relative_time_col] <= max_period)]
    
    # Create event time dummies
    # Need to create event time columns with statsmodels-friendly names (no negative signs)
    event_periods = sorted(data[relative_time_col].dropna().unique())
    event_cols = []
    
    # Create a mapping from periods to column names
    period_to_col = {}
    
    for period in event_periods:
        # Create columns with a format that statsmodels can handle
        if period < 0:
            col_name = f"event_m{abs(int(period))}"  # m for minus (e.g., event_m1 for -1)
        else:
            col_name = f"event_p{int(period)}"  # p for plus (e.g., event_p1 for +1)
        
        # Create indicator variable
        data[col_name] = (data[relative_time_col] == period).astype(int)
        event_cols.append(col_name)
        period_to_col[period] = col_name
    
    # Omit reference period (usually -1) for identification
    reference_period = -1
    reference_col = period_to_col.get(reference_period)
    if reference_col in event_cols:
        event_cols.remove(reference_col)
    
    # Construct formula
    formula_parts = [outcome_col, "~"]
    formula_parts.append(" + ".join(event_cols))
    formula_parts.append("+ C(" + unit_col + ")")
    formula_parts.append("+ C(" + time_col + ")")
    
    if control_cols:
        formula_parts.append("+ " + " + ".join(control_cols))
    
    formula = " ".join(formula_parts)
    
    # Create and return the EventStudySpec object
    return EventStudySpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols,
        time_col=time_col,
        unit_col=unit_col,
        relative_time_col=relative_time_col,
        event_cols=event_cols,
        reference_period=reference_period,
        data=data,
        formula=formula
    )


@make_transformable
def create_staggered_did_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    treatment_time_col: Optional[str] = None,
    control_cols: Optional[List[str]] = None,
    pre_periods: int = 4  
) -> StaggeredDiDSpec:
    """
    Create a Staggered DiD specification.
    
    Follows the cohort-based approach for staggered treatment adoption.
    
    Args:
        data: DataFrame with outcome, treatment, time, and unit identifiers
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names
        time_col: Name of time column
        unit_col: Name of unit identifier column
        treatment_time_col: Name of treatment timing column
        control_cols: List of control variable columns
        pre_periods: Number of pre-treatment periods to include for testing parallel trends
        
    Returns:
        StaggeredDiDSpec object with specification information
    """
    # Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols,
        excluded_cols=[time_col, unit_col]
    )
    
    # Determine treatment cohorts
    if treatment_time_col is None:
        # Infer treatment timing from the data
        treatment_times = data[data[treatment_col] == 1].groupby(unit_col)[time_col].min()
        data['treatment_time'] = data[unit_col].map(treatment_times)
        treatment_time_col = 'treatment_time'
    
    # Identify cohorts based on treatment timing
    cohorts = sorted(data[treatment_time_col].dropna().unique())
    if len(cohorts) == 0:
        raise ValueError("No treatment cohorts identified. Ensure treated units have valid treatment times.")
    
    # Create cohort indicators
    for cohort in cohorts:
        cohort_col = f'cohort_{int(cohort)}'
        data[cohort_col] = (data[treatment_time_col] == cohort).astype(int)
    
    cohort_cols = [f'cohort_{int(cohort)}' for cohort in cohorts]
    
    # Get minimum cohort period for normalization
    min_cohort = min(int(cohort) for cohort in cohorts)
    
    # Create relative time variable
    data['relative_time'] = data[time_col] - data[treatment_time_col]
    
    # Create pre-treatment and post-treatment indicators by event time
    pre_period_cols = []
    post_cols = []
    
    # Add post indicator for each cohort
    for cohort in cohorts:
        post_col = f'post_{int(cohort)}'
        data[post_col] = (data[time_col] >= cohort).astype(int)
        post_cols.append(post_col)
    
    # Create event time dummies for pre-periods
    interaction_cols = []
    
    # Pre-treatment periods
    for period in range(1, pre_periods + 1):
        # Negative period (pre-treatment)
        event_col = f'event_m{period}'
        
        # Instead of looking for treated==1 in pre-periods (which is impossible),
        # identify units that will eventually be treated and are in the pre-period
        # "Eventually treated" means treatment_time_col is not NaN
        data[event_col] = ((data['relative_time'] == -period) & 
                          data[treatment_time_col].notna()).astype(int)
        
        pre_period_cols.append(event_col)
        interaction_cols.append(event_col)
    
    # Add post-treatment effects (one per cohort)
    for cohort in cohorts:
        interaction_col = f'effect_{int(cohort)}'
        data[interaction_col] = data[f'cohort_{int(cohort)}'] * data[f'post_{int(cohort)}']
        interaction_cols.append(interaction_col)
    
    # Construct formula
    formula = f"{outcome_col} ~ {' + '.join(interaction_cols)}"
    
    # Add unit and time fixed effects
    formula += f" + C({unit_col}) + C({time_col})"
    
    if control_cols:
        formula += f" + {' + '.join(control_cols)}"
    
    # Create and return specification
    return StaggeredDiDSpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        time_col=time_col,
        unit_col=unit_col,
        treatment_time_col=treatment_time_col,
        cohorts=list(cohorts),  # Convert numpy array to list if needed
        cohort_cols=cohort_cols,
        post_cols=post_cols,
        interaction_cols=interaction_cols,
        control_cols=control_cols,
        data=data,
        formula=formula
    )


def spec_constructor(spec: Any) -> str:
    """Convert any specification object to a constructor string representation.
    
    Args:
        spec: Any specification object (DiDSpec, CrossSectionalSpec, etc.)
        
    Returns:
        A string representation calling the constructor with appropriate arguments
    """
    # Get the class name for the constructor call
    class_name = spec.__class__.__name__
    class_module = spec.__class__.__module__
    
    # Gather attributes excluding DataFrame objects
    attrs = []
    for key, value in spec.__dict__.items():
        if isinstance(value, pd.DataFrame):
            # Just reference the data variable
            attrs.append(f"    {key}=data_PLACEHOLDER")
        elif isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            # Format list of strings more cleanly
            items = ", ".join([repr(x) for x in value])
            attrs.append(f"    {key}=[{items}]")
        elif isinstance(value, str):
            # Add quotes around string values
            attrs.append(f'    {key}="{value}"')
        elif value is None:
            # Handle None values
            attrs.append(f"    {key}=None")
        else:
            # For other types (booleans, numbers, etc.)
            attrs.append(f"    {key}={value}")

    # add import statement for the class
    

    # Return the constructor call as a string with proper indentation
    return f"{class_name}(\n" + ",\n".join(attrs) + "\n)"













# def create_continuous_treatment_specification(
#     df: pd.DataFrame, 
#     outcome_col: str = 'y', 
#     treatment_col: str = 'treat',
#     time_col: Optional[str] = None,
#     unit_col: Optional[str] = None,
#     is_did: bool = False,
#     post_col: Optional[str] = None,
#     control_cols: Optional[List[str]] = None
# ) -> ContinuousTreatmentSpec:
#     """
#     Create a specification for models with continuous treatment.
    
#     Args:
#         df: DataFrame with outcome and continuous treatment
#         outcome_col: Name of outcome column
#         treatment_col: Name of treatment column
#         time_col: Name of time column (for panel data)
#         unit_col: Name of unit identifier column (for panel data)
#         is_did: Whether to use DiD framework with continuous treatment
#         post_col: Name of post-treatment indicator column (for DiD)
#         control_cols: List of control variable columns
        
#     Returns:
#         ContinuousTreatmentSpec object with specification information
#     """
#     # Validate required columns
#     required_columns = [outcome_col, treatment_col]
#     if is_did:
#         required_columns.extend([time_col, unit_col])
#         if post_col is not None:
#             required_columns.append(post_col)
            
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
#     # Clean data
#     df = df.copy().dropna()
    
#     # If control_cols not specified, use all numeric columns except required ones
#     if control_cols is None:
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         control_cols = [col for col in numeric_cols if col not in required_columns]
    
#     # Construct formula for standard regression with continuous treatment
#     formula_parts = [outcome_col, "~", treatment_col]
    
#     if control_cols:
#         formula_parts.extend(["+ " + " + ".join(control_cols)])
        
#     formula = " ".join(formula_parts)
    
#     # Create specification
#     spec = ContinuousTreatmentSpec(
#         outcome_col=outcome_col,
#         treatment_col=treatment_col,
#         control_cols=control_cols,
#         data=df,
#         formula=formula,
#         time_col=time_col,
#         unit_col=unit_col,
#         post_col=post_col,
#         interaction_col='treat_post'
#     )
    
#     if is_did:
#         spec.type = "continuous_did"  # Override the type if needed
        
#     return spec
        