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
import re
from linearmodels import PanelOLS, RandomEffects, FirstDifferenceOLS, BetweenOLS
from pyautocausal.pipelines.library.synthdid.synthdid import synthdid_estimate
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.specifications import (
    BaseSpec,
    DiDSpec,
    EventStudySpec,
    StaggeredDiDSpec,
    SynthDIDSpec,
)
from pyautocausal.pipelines.library.base_estimator import format_statsmodels_result
from pyautocausal.pipelines.library.callaway_santanna import fit_callaway_santanna as cs_fit
from pyautocausal.pipelines.library.csdid.att_gt import ATTgt
from pyautocausal.pipelines.library.synthdid.vcov import synthdid_se
    

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
def fit_ols(spec: BaseSpec, weights: Optional[np.ndarray] = None) -> BaseSpec:
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
def fit_weighted_ols(spec: BaseSpec) -> BaseSpec:
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
    
    This function uses the new csdid module implementation for more robust and comprehensive results.
    
    Args:
        spec: A StaggeredDiDSpec object with data and column information
        
    Returns:
        StaggeredDiDSpec with fitted model
    """
    
    
    # Extract necessary information from spec
    data = spec.data
    outcome_col = spec.outcome_col
    time_col = spec.time_col
    unit_col = spec.unit_col
    treatment_time_col = spec.treatment_time_col
    control_cols = spec.control_cols if hasattr(spec, 'control_cols') and spec.control_cols else []
    formula = spec.formula

    # Prepare data for csdid format
    data_cs = data.copy()
    
    # Ensure never-treated units have 0 in treatment_time_col, not NaN
    # This is required for the Callaway & Sant'Anna estimator
    data_cs[treatment_time_col] = data_cs[treatment_time_col].fillna(0)

    # create a new column that is 1 if the unit is treated at any time
    data_cs["never_treated"] = data_cs.groupby(unit_col)[treatment_time_col].transform('max') == 0


    att_gt = ATTgt(
        yname=outcome_col,
        tname=time_col,
        idname=unit_col,
        gname=treatment_time_col,
        data=data_cs,
        control_group=['nevertreated'],
        xformla=None,
        panel=True,
        allow_unbalanced_panel=True,
        anticipation=0,
        cband=True,
        biters=1000,
        alp=0.05
    )
    
    # Fit the model
    att_gt.fit(est_method='dr', base_period='varying', bstrap=True)
    
    # Generate summary table
    att_gt.summ_attgt(n=4)
    
    # Create separate copies for each aggregation to avoid overwriting
    import copy
    
    # Compute aggregated treatment effects
    # Overall effect
    att_gt_overall = copy.deepcopy(att_gt)
    att_gt_overall.aggte(typec="simple", bstrap=True, cband=True)
    
    # Dynamic (event study) effects  
    att_gt_dynamic = copy.deepcopy(att_gt)
    att_gt_dynamic.aggte(typec="dynamic", bstrap=True, cband=True)
    
    # Group-specific effects
    att_gt_group = copy.deepcopy(att_gt)
    att_gt_group.aggte(typec="group", bstrap=True, cband=True)
    
    # Store all results in the spec
    spec.model = {
        'att_gt_object': att_gt,
        'overall_effect': att_gt_overall,
        'dynamic_effects': att_gt_dynamic,
        'group_effects': att_gt_group,
        'control_group': 'never_treated',
        'estimator': 'callaway_santanna_csdid'
    }
    
    return spec


@make_transformable
def fit_callaway_santanna_nyt_estimator(spec: StaggeredDiDSpec) -> StaggeredDiDSpec:
    """
    Wrapper function to fit the Callaway and Sant'Anna (2021) DiD estimator using not-yet-treated units as the control group.
    
    This function uses the new csdid module implementation for more robust and comprehensive results.
    
    Args:
        spec: A StaggeredDiDSpec object with data and column information
        
    Returns:
        StaggeredDiDSpec with fitted model
    """
    from pyautocausal.pipelines.library.csdid.att_gt import ATTgt
    
    # Extract necessary information from spec
    data = spec.data
    outcome_col = spec.outcome_col
    time_col = spec.time_col
    unit_col = spec.unit_col
    treatment_time_col = spec.treatment_time_col
    control_cols = spec.control_cols if hasattr(spec, 'control_cols') and spec.control_cols else []
    
    # Prepare data for csdid format
    data_cs = data.copy()
    
    # csdid expects different data format:
    # 1. A time-invariant treatment indicator (treat: 1 if ever treated, 0 if never)
    # 2. first.treat column with the first treatment period for each unit
    
    # Check if first.treat already exists in the data
    if 'first.treat' not in data_cs.columns:
        # Create first.treat from the time-varying treat column
        # For each unit, find the first period where treat == 1
        first_treat_periods = data_cs[data_cs['treat'] == 1].groupby(unit_col)[time_col].min()
        
        # Create first.treat column - merge back to get first treatment time for each unit
        data_cs = data_cs.merge(
            first_treat_periods.reset_index().rename(columns={time_col: 'first.treat'}),
            on=unit_col, how='left'
        )
        
        # Units that never get treated should have 0 or NaN in first.treat
        data_cs['first.treat'] = data_cs['first.treat'].fillna(0)
    
    # Create time-invariant treatment indicator
    # 1 if unit is ever treated, 0 if never treated
    ever_treated_units = data_cs[data_cs['treat'] == 1][unit_col].unique()
    data_cs['treat_indicator'] = data_cs[unit_col].isin(ever_treated_units).astype(int)
    
    # Use first.treat as the group variable for csdid
    # csdid expects never-treated units to have gname = 0, not NaN
    data_cs['gname'] = data_cs['first.treat'].copy()
    
    # Create formula for controls if they exist
    # Note: Temporarily disabling control variables due to csdid internal data processing issues
    xformla = None
    if control_cols:
        print(f"Note: Control variables {control_cols} are available but temporarily disabled due to csdid module compatibility issues.")
    
    # Initialize the ATTgt object with not-yet-treated control group
    att_gt = ATTgt(
        yname=outcome_col,
        tname=time_col,
        idname=unit_col,
        gname='gname',
        data=data_cs,
        control_group=['notyettreated'],
        xformla=xformla,
        panel=True,
        allow_unbalanced_panel=True,
        anticipation=0,
        cband=True,
        biters=1000,
        alp=0.05
    )
    
    # Fit the model
    att_gt.fit(est_method='dr', base_period='varying', bstrap=True)
    
    # Generate summary table
    att_gt.summ_attgt(n=4)
    
    # Compute aggregated treatment effects
    # Overall effect
    att_gt_overall = att_gt.aggte(typec="simple", bstrap=True, cband=True)
    
    # Dynamic (event study) effects
    att_gt_dynamic = att_gt.aggte(typec="dynamic", bstrap=True, cband=True)
    
    # Group-specific effects
    att_gt_group = att_gt.aggte(typec="group", bstrap=True, cband=True)
    
    # Store all results in the spec
    spec.model = {
        'att_gt_object': att_gt,
        'overall_effect': att_gt_overall,
        'dynamic_effects': att_gt_dynamic,
        'group_effects': att_gt_group,
        'control_group': 'not_yet_treated',
        'estimator': 'callaway_santanna_csdid'
    }
    
    return spec


@make_transformable
def fit_synthdid_estimator(spec) -> object:
    """
    Fit a Synthetic Difference-in-Differences estimator.
    
    Args:
        spec: A SynthDIDSpec object with data and matrix information
        
    Returns:
        SynthDIDSpec with fitted model (SynthDIDEstimate object)
    """
    
    
    # Extract matrices from spec
    Y = spec.Y
    N0 = spec.N0
    T0 = spec.T0
    X = None  # TODO: Allow to use covariates that are not the matching variables
    
    # Fit the synthetic DiD model
    # Fit the model and get point estimate
    estimate = synthdid_estimate(Y, N0, T0, X=X)
    
    # Calculate standard error using placebo method
    se_result = synthdid_se(estimate, method='placebo')
    se = se_result['se']
    
    # Calculate confidence intervals
    ci_lower = float(estimate) - 1.96 * se
    ci_upper = float(estimate) + 1.96 * se
    
    print(f"Point estimate: {float(estimate):.2f}")
    print(f"95% CI ({ci_lower:.2f}, {ci_upper:.2f})")

    # Store results
    spec.model = estimate  # Store the actual SynthDIDEstimate object
    
    return spec


@make_transformable
def fit_panel_ols(spec: BaseSpec) -> BaseSpec:
    """
    Fit Panel OLS regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted PanelOLS model
    """
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index(['id_unit', 't'])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    formula = spec.formula
    
    # Set up clustering (default to entity clustering)
    cluster_config = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Fit model using from_formula with special effects variables
    model = PanelOLS.from_formula(formula, data_indexed, drop_absorbed=True, check_rank=False)
    
    result = model.fit(**cluster_config)
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_did_panel(spec: BaseSpec) -> BaseSpec:
    """
    Fit Difference-in-Differences using Panel OLS with entity and time fixed effects.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted PanelOLS model for DiD
    """
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index(['id_unit', 't'])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    formula = spec.formula
    
    # Set up clustering for DiD (cluster by entity)
    cluster_config = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Fit model with entity and time effects (standard DiD setup)
    model = PanelOLS.from_formula(formula, data_indexed, drop_absorbed=True, check_rank=False)
    
    result = model.fit(**cluster_config)
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)

    return spec


@make_transformable
def fit_random_effects(spec: BaseSpec) -> BaseSpec:
    """
    Fit Random Effects regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted RandomEffects model
    """
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index(['id_unit', 't'])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    # Note: RandomEffects doesn't use EntityEffects/TimeEffects syntax, 
    # so we need to clean those if present
    formula = spec.formula
    
    # For RandomEffects, remove EntityEffects and TimeEffects if present
    # since it handles random effects differently
    formula = formula.replace("+ EntityEffects", "").replace("+ TimeEffects", "")
    formula = formula.replace("EntityEffects +", "").replace("TimeEffects +", "")
    formula = formula.replace("EntityEffects", "").replace("TimeEffects", "")
    
    # Clean up any resulting double spaces or trailing/leading operators
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\+\s*\+', '+', formula)
    formula = re.sub(r'~\s*\+', '~', formula)
    formula = re.sub(r'\+\s*$', '', formula)
    formula = formula.strip()
    
    # Fit model
    model = RandomEffects.from_formula(formula, data_indexed)
    result = model.fit(cov_type='robust')
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_first_difference(spec: BaseSpec) -> BaseSpec:
    """
    Fit First Difference regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted FirstDifferenceOLS model
    """
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index(['id_unit', 't'])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    # Note: FirstDifferenceOLS doesn't use EntityEffects/TimeEffects and cannot include constants
    formula = spec.formula
    
    # For FirstDifferenceOLS, remove EntityEffects, TimeEffects, and constants
    formula = formula.replace("+ EntityEffects", "").replace("+ TimeEffects", "")
    formula = formula.replace("EntityEffects +", "").replace("TimeEffects +", "")
    formula = formula.replace("EntityEffects", "").replace("TimeEffects", "")
    formula = formula.replace("+ 1", "").replace("1 +", "")
    
    # Clean up any resulting double spaces or trailing/leading operators
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\+\s*\+', '+', formula)
    formula = re.sub(r'~\s*\+', '~', formula)
    formula = re.sub(r'\+\s*$', '', formula)
    formula = formula.strip()
    
    # Set up clustering
    cluster_config = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Fit model
    model = FirstDifferenceOLS.from_formula(formula, data_indexed)
    result = model.fit(**cluster_config)
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_between_estimator(spec: BaseSpec) -> BaseSpec:
    """
    Fit Between estimator regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted BetweenOLS model
    """
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index(['id_unit', 't'])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    # Note: BetweenOLS doesn't use EntityEffects/TimeEffects syntax
    formula = spec.formula
    
    # For BetweenOLS, remove EntityEffects and TimeEffects if present
    formula = formula.replace("+ EntityEffects", "").replace("+ TimeEffects", "")
    formula = formula.replace("EntityEffects +", "").replace("TimeEffects +", "")
    formula = formula.replace("EntityEffects", "").replace("TimeEffects", "")
    
    # Clean up any resulting double spaces or trailing/leading operators
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\+\s*\+', '+', formula)
    formula = re.sub(r'~\s*\+', '~', formula)
    formula = re.sub(r'\+\s*$', '', formula)
    formula = formula.strip()
    
    # Fit model
    model = BetweenOLS.from_formula(formula, data_indexed)
    result = model.fit(cov_type='robust')
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_hainmueller_synth_estimator(spec: SynthDIDSpec) -> SynthDIDSpec:
    """
    Fit a Hainmueeller Synthetic Control estimator using the SyntheticControlMethods package.
    
    Args:
        spec: A SynthDIDSpec object with data and matrix information
        
    Returns:
        SynthDIDSpec with fitted model (Synth object)
    """
    from SyntheticControlMethods import Synth
    
    # Extract information from spec
    data = spec.data.copy()
    outcome_col = spec.outcome_col
    time_col = spec.time_col
    unit_col = spec.unit_col
    treatment_col = spec.treatment_cols[0]
    
    # Find the treated unit and treatment period
    treated_units = data[data[treatment_col] == 1][unit_col].unique()

    treated_unit = treated_units[0]
    
    treatment_period = int(data[data[treatment_col] == 1][time_col].min())
    
    # Prepare control columns to exclude (treatment column and any non-numeric columns)
    exclude_columns = [treatment_col]
    for col in data.columns:
        if col not in [outcome_col, unit_col, time_col] and not pd.api.types.is_numeric_dtype(data[col]):
            exclude_columns.append(col)
    
    # Fit Synthetic Control using the SyntheticControlMethods package
    sc = Synth(
        dataset=data,
        outcome_var=outcome_col,
        id_var=unit_col, 
        time_var=time_col,
        treatment_period=treatment_period,
        treated_unit=treated_unit,
        n_optim=5,
        pen="auto",
        exclude_columns=exclude_columns,
        random_seed=42
    )
    
    # Print results for notebook display
    print(f"\nHainmueeller Synthetic Control Results:")
    print(f"Weight DataFrame:")
    print(sc.original_data.weight_df)
    print(f"\nComparison DataFrame:")
    print(sc.original_data.comparison_df.head())
    if hasattr(sc.original_data, 'pen'):
        print(f"\nPenalty parameter: {sc.original_data.pen}")
    
    # Store the Synth object in spec
    spec.hainmueller_model = sc
    
    return spec



@make_transformable
def fit_hainmueller_placebo_test(spec: SynthDIDSpec, n_placebo: int = 1) -> SynthDIDSpec:
    """
    Perform in-space  and in-time placebo test for Hainmueeller Synthetic Control method.
    """
    
    spec.hainmueller_model.in_space_placebo(n_placebo)
    
    return spec
    




