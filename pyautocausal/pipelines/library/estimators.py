"""
Estimator functions for causal inference models.
"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import patsy
from statsmodels.base.model import Results
import copy

from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.specifications import (
    BaseSpec,
    DiDSpec,
    EventStudySpec,
    StaggeredDiDSpec,
)
from pyautocausal.pipelines.library.base_estimator import format_statsmodels_result
from pyautocausal.pipelines.library.callaway_santanna import fit_callaway_santanna as cs_fit


def create_model_matrices(spec: BaseSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create model matrices from a specification using patsy.
    
    Args:
        spec: A specification dataclass with data and formula
        
    Returns:
        Tuple of (y, X) arrays for modeling
    """
    data = spec.data
    formula = spec.formula
    
    # Parse the formula into outcome and predictors
    outcome_expr, predictors_expr = formula.split('~', 1)
    outcome_expr = outcome_expr.strip()
    predictors_expr = predictors_expr.strip()
    
    # Create design matrices
    y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
    
    return y, X


def extract_treatment_from_design(X: pd.DataFrame, treatment_col: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract treatment variable from design matrix and return treatment and controls separately.
    
    Args:
        X: Design matrix containing treatment and control variables
        treatment_col: Name of the treatment column (in original data)
        
    Returns:
        Tuple of (treatment array, control design matrix)
    """
    # Look for columns that start with the treatment column name
    # This handles Patsy's transformations while avoiding partial matches
    treatment_cols = [col for col in X.columns if col.startswith(treatment_col)]
    
    if not treatment_cols:
        raise ValueError(f"Treatment column '{treatment_col}' not found in design matrix")
    
    # If there are multiple matching columns (e.g., interaction terms)
    if len(treatment_cols) > 1:
        # Prefer exact match if available
        exact_matches = [col for col in treatment_cols if col == treatment_col]
        if exact_matches:
            treatment_col_name = exact_matches[0]
        else:
            # Otherwise use the first match
            treatment_col_name = treatment_cols[0]
    else:
        treatment_col_name = treatment_cols[0]
    
    # Extract treatment
    d = X[treatment_col_name].values
    
    # Create control matrix without the treatment
    X_controls = X.drop(columns=treatment_col_name)
    
    return d, X_controls


@make_transformable
def fit_ols(spec: BaseSpec, weights: Optional[np.ndarray] = None) -> Results:
    """
    Estimate treatment effect using OLS regression.
    
    Args:
        spec: A specification dataclass with data and formula
        weights: Optional sample weights for weighted regression
        
    Returns:
        Specification with model field set to the fitted model
    """
    # We can still use the formula interface directly for OLS
    data = spec.data
    formula = spec.formula
    
    
    if weights is not None:
        model = sm.WLS.from_formula(formula, data=data, weights=weights).fit(cov_type='HC1')
    else:
        model = sm.OLS.from_formula(formula, data=data).fit(cov_type='HC1')
    
    # Set the model field (ensure all spec types handle this correctly)
    spec.model = model
            
    return spec


@make_transformable
def format_regression_results(model_result: Results) -> str:
    """
    Format regression model results as a readable string.
    
    Args:
        model_result: Statsmodels regression results object
        
    Returns:
        Formatted string with model summary
    """
    return format_statsmodels_result(model_result)


@make_transformable
def fit_weighted_ols(spec: BaseSpec) -> Results:
    """
    Estimate treatment effect using Weighted OLS regression.
    
    Args:
        spec: A specification dataclass with data and formula
        weights: Sample weights for the regression
        
    Returns:
        Fitted statsmodels fit_weighted_ols model results
    """
    if "weights" in spec.data.columns:
        weights = spec.data['weights']
    else:
        raise ValueError("Weights must be provided for weighted OLS")
    
    # Ensure weights are valid (no NaN, inf, or zero sum)
    if np.isnan(weights).any() or np.isinf(weights).any() or np.sum(weights) == 0:
        # In case of invalid weights, use uniform weights as fallback
        weights = np.ones_like(weights) / len(weights)
        
    # Ensure weights are properly scaled
    if np.sum(weights) != 1.0 and np.sum(weights) != 0.0:
        weights = weights / np.sum(weights)
        
    return fit_ols(spec, weights)


@make_transformable
def fit_double_lasso(
    spec: BaseSpec,
    alpha: float = 0.1,
    cv_folds: int = 5
) -> Results:
    """
    Estimate treatment effect using Double/Debiased Lasso.
    
    Implementation follows the algorithm from:
    Chernozhukov et al. (2018) - Double/Debiased Machine Learning
    
    Args:
        spec: A specification dataclass with data and column information
        alpha: Regularization strength for Lasso
        cv_folds: Number of cross-validation folds
            
    Returns:
        Fitted OLS model for the final stage regression
    """
    # Use patsy to create design matrices
    y_df, X_df = create_model_matrices(spec)
    y = y_df.values.ravel()  # Convert to 1D array
    
    # Extract treatment and controls
    d, X_controls = extract_treatment_from_design(X_df, spec.treatment_cols[0])
    
    # Check if we have control variables
    if X_controls.shape[1] == 0:
        raise ValueError("Control variables are required for double lasso")
    
    # Initialize cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Arrays to store residuals
    y_resid = np.zeros_like(y)
    d_resid = np.zeros_like(d)
    
    # Cross-fitting
    for train_idx, test_idx in kf.split(X_controls):
        # Split data
        X_train, X_test = X_controls.iloc[train_idx], X_controls.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]
        
        # First stage: outcome equation
        lasso_y = Lasso(alpha=alpha, random_state=42)
        lasso_y.fit(X_train, y_train)
        y_pred = lasso_y.predict(X_test)
        y_resid[test_idx] = y_test - y_pred
        
        # First stage: treatment equation
        lasso_d = Lasso(alpha=alpha, random_state=42)
        lasso_d.fit(X_train, d_train)
        d_pred = lasso_d.predict(X_test)
        d_resid[test_idx] = d_test - d_pred
    
    # Second stage: treatment effect estimation
    final_model = sm.OLS(
        y_resid,
        sm.add_constant(d_resid)
    ).fit(cov_type='HC1')
    
    return final_model


@make_transformable
def fit_callaway_santanna_estimator(spec: StaggeredDiDSpec) -> StaggeredDiDSpec:
    """
    Wrapper function to fit the Callaway and Sant'Anna (2021) DiD estimator using never-treated units as the control group.
    
    This function is a simple wrapper around the implementation in the callaway_santanna module,
    designed to maintain consistency with other estimator functions.
    
    Args:
        spec: A StaggeredDiDSpec object with data and column information
        
    Returns:
        StaggeredDiDSpec with fitted model
    """
    return cs_fit(spec, control_group="never_treated")


@make_transformable
def fit_callaway_santanna_nyt_estimator(spec: StaggeredDiDSpec) -> StaggeredDiDSpec:
    """
    Wrapper function to fit the Callaway and Sant'Anna (2021) DiD estimator using not-yet-treated units as the control group.
    
    This function is a simple wrapper around the implementation in the callaway_santanna module,
    designed to maintain consistency with other estimator functions.
    
    Args:
        spec: A StaggeredDiDSpec object with data and column information
        
    Returns:
        StaggeredDiDSpec with fitted model
    """
    return cs_fit(spec, control_group="not_yet_treated")


@make_transformable
def fit_synthdid_estimator(spec) -> object:
    """
    Fit a Synthetic Difference-in-Differences estimator.
    
    Args:
        spec: A SynthDIDSpec object with data and matrix information
        
    Returns:
        SynthDIDSpec with fitted model (SynthDIDEstimate object)
    """
    from pyautocausal.pipelines.library.synthdid_py.synthdid import synthdid_estimate
    
    # Extract matrices from spec
    Y = spec.Y
    N0 = spec.N0
    T0 = spec.T0
    X = spec.X
    
    # Fit the synthetic DiD model
    estimate = synthdid_estimate(Y, N0, T0, X=X)
    
    # Store the estimate in the spec
    spec.model = estimate
    
    return spec

