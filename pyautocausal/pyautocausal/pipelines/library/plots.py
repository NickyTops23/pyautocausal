"""
Plot functions for causal inference visualizations
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
import re
import warnings

from pyautocausal.pipelines.library.specifications import (
    BaseSpec, DiDSpec, StaggeredDiDSpec, EventStudySpec
)
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.callaway_santanna import CSResults
from pyautocausal.persistence.output_config import OutputConfig, OutputType


def _get_confidence_intervals(model, confidence_level: float):
    """
    Helper function to get confidence intervals from both statsmodels and linearmodels results.
    
    Args:
        model: Fitted model (statsmodels or linearmodels result)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        DataFrame with confidence intervals with standardized column names
    """
    try:
        # Try linearmodels API first (uses level parameter)
        conf_int = model.conf_int(level=confidence_level)
        # Standardize column names if they're not already 'lower' and 'upper'
        if list(conf_int.columns) == [0, 1]:
            conf_int.columns = ['lower', 'upper']
        return conf_int
    except TypeError:
        try:
            # Fall back to statsmodels API (uses alpha parameter)
            alpha = 1 - confidence_level
            conf_int = model.conf_int(alpha=alpha)
            # Standardize column names if they're not already 'lower' and 'upper'
            if list(conf_int.columns) == [0, 1]:
                conf_int.columns = ['lower', 'upper']
            return conf_int
        except Exception as e:
            raise ValueError(f"Could not get confidence intervals from model: {e}")


def _extract_cs_results(cs_model: CSResults) -> pd.DataFrame:
    """Extract event study results from Callaway & Sant'Anna model."""
    cs_event_results = cs_model.summary[cs_model.summary['type'] == 'event_time'].copy()
    
    if cs_event_results.empty:
        raise ValueError("No event-time results found in the Callaway and Sant'Anna model")
    
    return pd.DataFrame({
        'period': cs_event_results['event_time'].tolist(),
        'coef': cs_event_results['att'].tolist(),
        'lower': cs_event_results['lower_ci'].tolist(),
        'upper': cs_event_results['upper_ci'].tolist()
    })


def _extract_event_study_results(spec: EventStudySpec, params: pd.Series, conf_int: pd.DataFrame) -> pd.DataFrame:
    """Extract event study results from EventStudySpec."""
    periods = []
    coeffs = []
    lower_ci = []
    upper_ci = []
    
    event_pattern_minus = re.compile(r'event_m(\d+)')  # For negative periods: event_m1 = -1
    event_pattern_plus = re.compile(r'event_p(\d+)')   # For positive periods: event_p1 = +1
    
    # Process event columns in chronological order
    for col in sorted(spec.event_cols):
        # Check for negative period (m prefix)
        match_minus = event_pattern_minus.match(col)
        if match_minus and col in params.index:
            period = -int(match_minus.group(1))  # Convert back to negative number
            periods.append(period)
            coeffs.append(params[col])
            lower_ci.append(conf_int.loc[col, 'lower'])
            upper_ci.append(conf_int.loc[col, 'upper'])
            continue
            
        # Check for positive/zero period (p prefix)
        match_plus = event_pattern_plus.match(col)
        if match_plus and col in params.index:
            period = int(match_plus.group(1))  # Already positive
            periods.append(period)
            coeffs.append(params[col])
            lower_ci.append(conf_int.loc[col, 'lower'])
            upper_ci.append(conf_int.loc[col, 'upper'])
    
    # Reference period should have zero effect by definition
    if spec.reference_period not in periods:
        periods.append(spec.reference_period)
        coeffs.append(0)
        lower_ci.append(0)
        upper_ci.append(0)
    
    return pd.DataFrame({
        'period': periods,
        'coef': coeffs,
        'lower': lower_ci,
        'upper': upper_ci
    })


def _extract_staggered_did_results(spec: StaggeredDiDSpec, params: pd.Series, conf_int: pd.DataFrame) -> pd.DataFrame:
    """Extract event study results from StaggeredDiDSpec."""
    if not (len(spec.interaction_cols) > 0 and any(col in params.index for col in spec.interaction_cols)):
        raise ValueError("No interaction columns found in the StaggeredDiD model parameters")
    
    periods = []
    coeffs = []
    lower_ci = []
    upper_ci = []
    
    # Extract pre-period effects directly from event_m* columns
    event_pattern_minus = re.compile(r'event_m(\d+)')
    for param_name in sorted(params.index):
        match = event_pattern_minus.match(param_name)
        if match and param_name in params.index:
            period = -int(match.group(1))  # Convert back to negative number
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    # Add treatment period (time 0) as reference with 0 effect
    periods.append(0)
    coeffs.append(0.0)
    lower_ci.append(0.0)
    upper_ci.append(0.0)
    
    # Extract post-treatment effects
    effect_pattern = re.compile(r'effect_(\d+)')
    
    # Get minimum cohort period for normalization
    min_cohort = min(int(cohort) for cohort in spec.cohorts)
    
    # Process each effect parameter
    for param_name in sorted(params.index):
        match = effect_pattern.match(param_name)
        if match:
            # Get the period number from the effect name
            raw_period = int(match.group(1))
            
            # Transform to relative period (relative to first treatment time)
            # If effect_5, effect_6, etc. make them relative time 1, 2, etc.
            relative_period = raw_period - min_cohort + 1
            
            # Add the effect parameters
            periods.append(relative_period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    return pd.DataFrame({
        'period': periods,
        'coef': coeffs,
        'lower': lower_ci,
        'upper': upper_ci
    })


def _extract_generic_results(params: pd.Series, conf_int: pd.DataFrame) -> pd.DataFrame:
    """Extract results from generic model with event patterns or treat_post fallback."""
    # Look for any event_* columns in parameters as fallback
    event_pattern = re.compile(r'event_(-?\d+)')
    periods = []
    coeffs = []
    lower_ci = []
    upper_ci = []
    
    for param_name in params.index:
        match = event_pattern.match(param_name)
        if match:
            period = int(match.group(1))
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    if not periods:
        # Final fallback - if treat_post exists, create basic DiD plot
        if 'treat_post' in params.index:
            periods = [-1, 1]
            coeffs = [0, params['treat_post']]
            lower_ci = [0, conf_int.loc['treat_post', 'lower']]
            upper_ci = [0, conf_int.loc['treat_post', 'upper']]
        else:
            raise ValueError("Could not identify any event time indicators in the model parameters")
    
    return pd.DataFrame({
        'period': periods,
        'coef': coeffs,
        'lower': lower_ci,
        'upper': upper_ci
    })


def _create_event_plot(results_df: pd.DataFrame, 
                      confidence_level: float = 0.95,
                      figsize: tuple = (12, 8),
                      title: str = "Event Study Plot",
                      xlabel: str = "Event Time",
                      ylabel: str = "Coefficient Estimate",
                      reference_line_color: str = "gray",
                      reference_line_style: str = "--",
                      effect_color: str = "blue",
                      marker: str = "o") -> plt.Figure:
    """Create the actual event study plot from standardized results DataFrame."""
    # Sort by period
    results_df = results_df.sort_values('period')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot coefficients with markers only (no connecting lines)
    ax.plot(results_df['period'], results_df['coef'], 
            marker=marker, 
            color=effect_color, 
            linestyle='none',  # This removes the connecting lines
            label='Coefficient')
    
    # Add error bars (whiskers) with same color as points
    yerr = [results_df['coef'] - results_df['lower'], results_df['upper'] - results_df['coef']]
    ax.errorbar(results_df['period'], results_df['coef'], 
               yerr=yerr, 
               fmt='none',  # No additional markers
               ecolor=effect_color,  # Same color as points
               elinewidth=2,
               capsize=5,
               label=f'{int(confidence_level*100)}% CI')
    
    # Add zero reference line
    ax.axhline(y=0, color=reference_line_color, linestyle=reference_line_style)
    
    # Add vertical line at period 0 (treatment time)
    ax.axvline(x=0, color=reference_line_color, linestyle=reference_line_style, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


@make_transformable
def event_study_plot(spec: Union[DiDSpec, StaggeredDiDSpec, EventStudySpec], 
                    confidence_level: float = 0.95,
                    figsize: tuple = (12, 8),
                    title: str = "Event Study Plot",
                    xlabel: str = "Event Time",
                    ylabel: str = "Coefficient Estimate",
                    reference_line_color: str = "gray",
                    reference_line_style: str = "--",
                    effect_color: str = "blue",
                    confidence_color: str = "lightblue",
                    marker: str = "o") -> plt.Figure:
    """
    Create an event study plot from a specification with a fitted model.
    
    Args:
        spec: A specification object with fitted model
        confidence_level: Confidence level for intervals (default: 0.95)
        figsize: Figure size as (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        reference_line_color: Color for the zero reference line
        reference_line_style: Line style for the zero reference line
        effect_color: Color for the effect point estimates
        confidence_color: Color for the confidence intervals
        marker: Marker style for point estimates
        
    Returns:
        Matplotlib figure with the event study plot
    """
    # Check if model exists
    if spec.model is None:
        raise ValueError("Specification must contain a fitted model")
    
    # Extract results based on model type
    if isinstance(spec.model, CSResults):
        results_df = _extract_cs_results(spec.model)
        title = title + " (Callaway & Sant'Anna)"
        ylabel = "ATT Estimate"
    else:
        # For standard OLS models
        params = spec.model.params
        conf_int = _get_confidence_intervals(spec.model, confidence_level)
        
        # Extract results based on specification type
        if isinstance(spec, EventStudySpec):
            results_df = _extract_event_study_results(spec, params, conf_int)
        elif isinstance(spec, StaggeredDiDSpec):
            results_df = _extract_staggered_did_results(spec, params, conf_int)
        else:
            results_df = _extract_generic_results(params, conf_int)
    
    # Create and return the plot
    return _create_event_plot(
        results_df=results_df,
        confidence_level=confidence_level,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        reference_line_color=reference_line_color,
        reference_line_style=reference_line_style,
        effect_color=effect_color,
        marker=marker
    )


def create_did_plot(spec: DiDSpec, 
                   figsize: tuple = (12, 8),
                   title: str = "Difference-in-Differences Plot",
                   xlabel: str = "Time",
                   ylabel: str = "Outcome",
                   treatment_color: str = "red",
                   control_color: str = "blue",
                   confidence_level: float = 0.95) -> plt.Figure:
    """
    Create a Difference-in-Differences plot showing pre and post trends.
    
    Args:
        spec: DiDSpec object with data
        figsize: Figure size as (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        treatment_color: Color for the treatment group line
        control_color: Color for the control group line
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Matplotlib figure with the DiD plot
    """
    # Check if data exists
    if spec.data is None:
        raise ValueError("DiD specification must contain data")
    
    # Extract column names
    data = spec.data
    time_col = spec.time_col
    unit_col = spec.unit_col
    outcome_col = spec.outcome_col
    treatment_col = spec.treatment_cols[0]  # Use first treatment column
    post_col = spec.post_col
    
    # Aggregate data by time and treatment status with standard error
    grouped = data.groupby([time_col, treatment_col])
    means = grouped[outcome_col].mean().reset_index()
    
    # Calculate standard errors for whiskers
    std_errors = grouped[outcome_col].agg(lambda x: x.std() / np.sqrt(len(x))).reset_index()
    
    # Join means and standard errors
    result = pd.merge(means, std_errors, on=[time_col, treatment_col], suffixes=('_mean', '_se'))
    
    # Calculate confidence interval multiplier
    import scipy.stats as stats
    ci_multiplier = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Calculate confidence intervals
    result['ci_lower'] = result[f'{outcome_col}_mean'] - ci_multiplier * result[f'{outcome_col}_se']
    result['ci_upper'] = result[f'{outcome_col}_mean'] + ci_multiplier * result[f'{outcome_col}_se']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot treatment group with error bars (no connecting lines)
    treated_data = result[result[treatment_col] == 1]
    ax.errorbar(treated_data[time_col], treated_data[f'{outcome_col}_mean'], 
              yerr=[treated_data[f'{outcome_col}_mean'] - treated_data['ci_lower'], 
                    treated_data['ci_upper'] - treated_data[f'{outcome_col}_mean']],
              fmt='o', color=treatment_color, ecolor=treatment_color, alpha=0.7,
              elinewidth=1.5, capsize=4, label='Treatment Group',
              linestyle='none')  # Remove connecting lines
    
    # Plot control group with error bars (no connecting lines)
    control_data = result[result[treatment_col] == 0]
    ax.errorbar(control_data[time_col], control_data[f'{outcome_col}_mean'], 
              yerr=[control_data[f'{outcome_col}_mean'] - control_data['ci_lower'], 
                    control_data['ci_upper'] - control_data[f'{outcome_col}_mean']],
              fmt='s', color=control_color, ecolor=control_color, alpha=0.7,
              elinewidth=1.5, capsize=4, label='Control Group',
              linestyle='none')  # Remove connecting lines
    
    # Find treatment timing (first period where post==1 for treated units)
    first_post_period = data[(data[treatment_col] == 1) & (data[post_col] == 1)][time_col].min()
    
    # Add vertical line for treatment timing
    if pd.notna(first_post_period):
        ax.axvline(x=first_post_period, color='gray', linestyle='--', 
                  label=f'Treatment Time ({first_post_period})')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

@make_transformable
def synthdid_plot(spec, output_config: Optional[OutputConfig] = None, **kwargs) -> plt.Figure:
    """
    Create a synthetic difference-in-differences plot.
    
    Args:
        spec: A SynthDIDSpec object with fitted model
        output_config: Configuration for saving the plot
        **kwargs: Additional arguments passed to plot_synthdid
        
    Returns:
        matplotlib figure object
    """
    from pyautocausal.pipelines.library.synthdid_py.plot import plot_synthdid
    
    if not hasattr(spec, 'model') or spec.model is None:
        raise ValueError("Specification must have a fitted model")
    
    # Create the plot
    fig, ax = plot_synthdid(spec.model, **kwargs)
    
    # Save if output config provided
    if output_config:
        if output_config.output_type == OutputType.PNG:
            fig.savefig(f"{output_config.output_filename}.png", 
                       dpi=300, bbox_inches='tight')
        elif output_config.output_type == OutputType.PDF:
            fig.savefig(f"{output_config.output_filename}.pdf", 
                       bbox_inches='tight')
    
    return fig
