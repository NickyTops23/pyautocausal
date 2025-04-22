import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from .base_weighter import BaseWeighter
from pyautocausal.persistence.parameter_mapper import make_transformable
from .specifications import DiDSpec


@make_transformable
def compute_synethetic_control_weights(spec: DiDSpec,
                    unit_col: str = 'id_unit', 
                    time_col: str = 't',
                    outcome_col: str = 'y',
                        treatment_col: str = 'treat',
                        covariates: Optional[List[str]] = None,
                        pre_treatment_periods: Optional[List] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Compute synthetic control weights.
        
        Args:
            inputs: Dictionary containing DataFrame with units, time periods, outcome, and treatment
            unit_col: Column name for unit identifier
            time_col: Column name for time period
            outcome_col: Column name for outcome variable
            treatment_col: Column name for treatment indicator
            covariates: List of covariates to use (if None, uses outcome in pre-treatment periods)
            pre_treatment_periods: List of pre-treatment periods (if None, inferred from data)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with synthetic control information including data and weights
        """
        # Extract data from inputs
        if isinstance(spec, DiDSpec) and 'data' in spec:
            df = spec.data
        else:
            raise ValueError("Inputs must contain a 'data' field with a DataFrame")
        
        unit_col = spec.unit_col
        time_col = spec.time_col
        outcome_col = spec.outcome_col
        
        # sort of hacky, think about how to elegantly handle multiple vs one treatment column
        treatment_col = spec.treatment_cols[0] 

        required_columns = [unit_col, time_col, outcome_col, treatment_col]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Identify treated unit(s)
        treated_units = df.loc[df[treatment_col] == 1, unit_col].unique()
        if len(treated_units) == 0:
            raise ValueError("No treated units found in data")
        if len(treated_units) > 1:
            raise ValueError("Synthetic control requires exactly one treated unit")
        
        treated_unit = treated_units[0]
        
        # Identify control periods for the treated unit
        control_periods = df[(df[unit_col] == treated_unit) & (df[treatment_col] == 0)][time_col].unique()
        
        # Check if we have other control units
        other_control_units = df[(df[unit_col] != treated_unit) & (df[treatment_col] == 0)][unit_col].unique()
        
        # If there are no other control units and no control periods for the treated unit, raise an error
        if len(other_control_units) == 0 and len(control_periods) == 0:
            raise ValueError("No control units or control periods found for synthetic control")
        
        # If treated unit is also a control unit in some periods, use those periods
        if len(control_periods) > 0:
            control_units = np.array([treated_unit])
        else:
            # Otherwise use other control units
            control_units = other_control_units
        
        # Determine pre-treatment periods
        if pre_treatment_periods is None:
            # Get all periods before treatment starts for the treated unit
            treat_start = df[(df[unit_col] == treated_unit) & (df[treatment_col] == 1)][time_col].min()
            pre_treatment_periods = sorted(df[df[time_col] < treat_start][time_col].unique())
        
        if len(pre_treatment_periods) == 0:
            raise ValueError("No pre-treatment periods identified")
        
        # Get pre-treatment outcomes for treated unit
        treated_pre = df[(df[unit_col] == treated_unit) & 
                         (df[time_col].isin(pre_treatment_periods))][outcome_col].values
        
        # Get pre-treatment outcomes for control units
        control_pre = np.zeros((len(control_units), len(pre_treatment_periods)))
        
        for i, unit in enumerate(control_units):
            unit_data = df[(df[unit_col] == unit) & 
                           (df[time_col].isin(pre_treatment_periods))][outcome_col].values
            control_pre[i, :] = unit_data
        
        # For single unit case, use all 1.0 for weights
        if len(control_units) == 1 and control_units[0] == treated_unit:
            unit_weights = np.ones(len(control_units))
        else:
            # For multiple control units, use equal weights
            unit_weights = np.ones(len(control_units)) / len(control_units)
        
        # Compute synthetic control as weighted average of control units
        synthetic_pre = unit_weights @ control_pre
        
        # Calculate fit metrics
        pre_rmse = np.sqrt(np.mean((treated_pre - synthetic_pre) ** 2))
        pre_mae = np.mean(np.abs(treated_pre - synthetic_pre))
        
        # Create full weights array for all units in the dataset
        all_units = df[unit_col].unique()
        all_weights = np.zeros(len(all_units))
        
        # Set weights for control units
        for i, unit in enumerate(control_units):
            unit_idx = np.where(all_units == unit)[0][0]
            all_weights[unit_idx] = unit_weights[i]
        
        # Create a new copy of inputs to avoid modifying the original
        result = inputs.copy()
        
        # Store weights in the result
        result['weights'] = all_weights
        
        # Create a diagnostic report to help with troubleshooting
        result['sc_diagnostics'] = {
            'treated_unit': treated_unit,
            'control_units': control_units.tolist(),
            'pre_treatment_periods': pre_treatment_periods,
            'pre_rmse': pre_rmse,
            'pre_mae': pre_mae,
            'num_units': len(all_units),
            'num_controls': len(control_units)
        }
        
        return result