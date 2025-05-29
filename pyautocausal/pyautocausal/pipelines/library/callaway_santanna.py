"""
Callaway and Sant'Anna (2021) Difference-in-Differences Estimator.

This module implements the Callaway and Sant'Anna (2021) Difference-in-Differences
estimator for staggered treatment adoption settings.

Reference:
Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods.
Journal of Econometrics, 225(2), 200-230.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.specifications import StaggeredDiDSpec, BaseSpec


@dataclass
class CSResults:
    """
    Results from the Callaway and Sant'Anna estimator.
    
    Attributes:
        att_dict: Dictionary of ATT estimates by cohort and time
        att_cohort: Average treatment effect on the treated by cohort
        att_event: Average treatment effect on the treated by event time
        att_overall: Overall average treatment effect on the treated
        summary: Summary dataframe of all ATT estimates
        standard_errors: Standard errors for ATT estimates
        data: Original data used for estimation
    """
    att_dict: Dict[Tuple[int, int], float]
    att_cohort: Dict[int, float]
    att_event: Dict[int, float]
    att_overall: float
    summary: pd.DataFrame
    standard_errors: Dict[str, Dict[Union[Tuple[int, int], int], float]]
    data: pd.DataFrame


def _compute_group_time_att(
    data: pd.DataFrame,
    cohort: int,
    time: int,
    outcome_col: str,
    time_col: str,
    cohort_col: str,
    unit_col: str,
    control_cols: Optional[List[str]] = None,
    control_group: str = "never_treated"
) -> Tuple[float, float]:
    """
    Compute the group-time average treatment effect on the treated (ATT).
    
    Args:
        data: DataFrame with data
        cohort: Treatment cohort (when units were treated)
        time: Time period for which to compute ATT
        outcome_col: Name of outcome variable column
        time_col: Name of time period column
        cohort_col: Name of cohort column
        unit_col: Name of unit ID column
        control_cols: List of control variable columns
        control_group: Type of control group to use ("never_treated" or "not_yet_treated")
        
    Returns:
        ATT estimate and standard error for the cohort-time pair
    """
    if control_cols is None:
        control_cols = []
    
    # Get units in this cohort
    cohort_units = data[data[cohort_col] == cohort][unit_col].unique()
    
    # Get control units based on the specified control group type
    if control_group == "never_treated":
        # Never-treated units (traditional control group)
        control_units = data[pd.isna(data[cohort_col])][unit_col].unique()
    elif control_group == "not_yet_treated":
        # Units not yet treated at time period 'time'
        later_cohorts = [c for c in data[cohort_col].dropna().unique() if c > time]
        control_units = data[data[cohort_col].isin(later_cohorts)][unit_col].unique()
        
        if len(control_units) == 0:
            # Fallback to never-treated if no not-yet-treated units
            control_units = data[pd.isna(data[cohort_col])][unit_col].unique()
    else:
        raise ValueError(f"Unknown control group type: {control_group}. Use 'never_treated' or 'not_yet_treated'.")
    
    # Filter to cohort and control units
    cohort_data = data[data[unit_col].isin(list(cohort_units) + list(control_units))].copy()
    
    # Get all time periods before treatment for this cohort
    pre_periods = [t for t in data[time_col].unique() if t < cohort]
    
    if not pre_periods:
        # No pre-periods available for this cohort
        return np.nan, np.nan
        
    # Use last pre-period as reference
    reference_period = max(pre_periods)
    
    # Filter to the reference period and current period
    period_data = cohort_data[cohort_data[time_col].isin([reference_period, time])].copy()
    
    # Create post indicator (using .loc to avoid SettingWithCopyWarning)
    period_data.loc[:, 'post'] = (period_data[time_col] == time).astype(int)
    
    # Create treatment indicator (1 for cohort, 0 for control)
    period_data.loc[:, 'treat'] = (period_data[cohort_col] == cohort).astype(int)
    
    # Create interaction term
    period_data.loc[:, 'treat_post'] = period_data['treat'] * period_data['post']
    
    # Prepare formula for DiD regression
    formula = f"{outcome_col} ~ treat + post + treat_post"
    if control_cols:
        formula += " + " + " + ".join(control_cols)
    
    # Fit DiD model
    model = sm.OLS.from_formula(formula, data=period_data).fit(cov_type='HC1')
    
    # Extract the ATT (coefficient on the interaction term)
    att = model.params['treat_post']
    se = model.bse['treat_post']
    
    return att, se


@make_transformable
def fit_callaway_santanna(
    spec: StaggeredDiDSpec,
    control_group: str = "never_treated"
) -> StaggeredDiDSpec:
    """
    Fit the Callaway and Sant'Anna (2021) Difference-in-Differences estimator.
    
    Args:
        spec: StaggeredDiDSpec object with data and column information
        control_group: Type of control group to use ("never_treated" or "not_yet_treated")
        
    Returns:
        Original spec with fitted cs_model attribute
    """
    # Extract data and columns from spec
    data = spec.data
    outcome_col = spec.outcome_col
    time_col = spec.time_col
    unit_col = spec.unit_col
    treatment_time_col = spec.treatment_time_col
    control_cols = spec.control_cols
    
    # Ensure we have the cohort columns
    cohort_col = treatment_time_col  # The treatment timing serves as cohort indicator
    
    # Get all time periods and cohorts
    time_periods = sorted(data[time_col].unique())
    cohorts = sorted([c for c in spec.cohorts if not pd.isna(c)])
    
    # Compute ATT for each cohort-time combination
    att_dict = {}
    se_dict = {}
    
    for cohort in cohorts:
        # For each cohort, compute ATT for all post-treatment periods
        post_periods = [t for t in time_periods if t >= cohort]
        
        for time in post_periods:
            # Compute ATT for this cohort-time pair
            att, se = _compute_group_time_att(
                data=data,
                cohort=cohort,
                time=time,
                outcome_col=outcome_col,
                time_col=time_col,
                cohort_col=cohort_col,
                unit_col=unit_col,
                control_cols=control_cols,
                control_group=control_group
            )
            
            # Store results
            att_dict[(cohort, time)] = att
            se_dict[(cohort, time)] = se
    
    # Compute cohort-specific ATTs (average across post-treatment periods)
    att_cohort = {}
    se_cohort = {}
    
    for cohort in cohorts:
        # Get all post-treatment periods for this cohort
        post_periods = [t for t in time_periods if t >= cohort]
        
        # Get ATTs for this cohort
        cohort_atts = [att_dict.get((cohort, t), np.nan) for t in post_periods]
        cohort_atts = [att for att in cohort_atts if not np.isnan(att)]
        
        if cohort_atts:
            # Compute simple average
            att_cohort[cohort] = np.mean(cohort_atts)
            
            # Approximate standard error
            se_values = [se_dict.get((cohort, t), np.nan) for t in post_periods]
            se_values = [se for se in se_values if not np.isnan(se)]
            se_cohort[cohort] = np.sqrt(np.mean([se**2 for se in se_values])) / np.sqrt(len(se_values))
    
    # Compute event-study ATTs (average across cohorts for same relative time)
    att_event = {}
    se_event = {}
    
    # Determine all possible event times
    event_times = []
    for cohort in cohorts:
        post_periods = [t for t in time_periods if t >= cohort]
        event_times.extend([t - cohort for t in post_periods])
    event_times = sorted(set(event_times))
    
    for event_time in event_times:
        # Get all (cohort, time) pairs for this event time
        event_pairs = []
        for cohort in cohorts:
            time = cohort + event_time
            if time in time_periods and time >= cohort:
                event_pairs.append((cohort, time))
        
        # Get ATTs for this event time
        event_atts = [att_dict.get(pair, np.nan) for pair in event_pairs]
        event_atts = [att for att in event_atts if not np.isnan(att)]
        
        if event_atts:
            # Compute simple average
            att_event[event_time] = np.mean(event_atts)
            
            # Approximate standard error
            se_values = [se_dict.get(pair, np.nan) for pair in event_pairs]
            se_values = [se for se in se_values if not np.isnan(se)]
            se_event[event_time] = np.sqrt(np.mean([se**2 for se in se_values])) / np.sqrt(len(se_values))
    
    # Compute overall ATT
    all_atts = [att for att in att_dict.values() if not np.isnan(att)]
    att_overall = np.mean(all_atts) if all_atts else np.nan
    
    # Approximate overall standard error
    all_ses = [se for se in se_dict.values() if not np.isnan(se)]
    se_overall = np.sqrt(np.mean([se**2 for se in all_ses])) / np.sqrt(len(all_ses)) if all_ses else np.nan
    
    # Create summary dataframe
    summary_rows = []
    
    # Add group-time ATTs
    for (cohort, time), att in att_dict.items():
        if not np.isnan(att):
            summary_rows.append({
                'type': 'group_time',
                'cohort': cohort,
                'time': time,
                'event_time': time - cohort,
                'att': att,
                'se': se_dict.get((cohort, time), np.nan),
                'lower_ci': att - 1.96 * se_dict.get((cohort, time), np.nan),
                'upper_ci': att + 1.96 * se_dict.get((cohort, time), np.nan)
            })
    
    # Add cohort ATTs
    for cohort, att in att_cohort.items():
        summary_rows.append({
            'type': 'cohort',
            'cohort': cohort,
            'time': np.nan,
            'event_time': np.nan,
            'att': att,
            'se': se_cohort.get(cohort, np.nan),
            'lower_ci': att - 1.96 * se_cohort.get(cohort, np.nan),
            'upper_ci': att + 1.96 * se_cohort.get(cohort, np.nan)
        })
    
    # Add event-time ATTs
    for event_time, att in att_event.items():
        summary_rows.append({
            'type': 'event_time',
            'cohort': np.nan,
            'time': np.nan,
            'event_time': event_time,
            'att': att,
            'se': se_event.get(event_time, np.nan),
            'lower_ci': att - 1.96 * se_event.get(event_time, np.nan),
            'upper_ci': att + 1.96 * se_event.get(event_time, np.nan)
        })
    
    # Add overall ATT
    summary_rows.append({
        'type': 'overall',
        'cohort': np.nan,
        'time': np.nan,
        'event_time': np.nan,
        'att': att_overall,
        'se': se_overall,
        'lower_ci': att_overall - 1.96 * se_overall,
        'upper_ci': att_overall + 1.96 * se_overall
    })
    
    # Create summary dataframe
    summary = pd.DataFrame(summary_rows)
    
    # Create standard errors dictionary
    standard_errors = {
        'group_time': se_dict,
        'cohort': se_cohort,
        'event_time': se_event,
        'overall': se_overall
    }
    
    # Create results object
    results = CSResults(
        att_dict=att_dict,
        att_cohort=att_cohort,
        att_event=att_event,
        att_overall=att_overall,
        summary=summary,
        standard_errors=standard_errors,
        data=data
    )
    
    # Store results in specification
    spec.model = results
    
    return spec


@make_transformable
def event_study_plot_callaway(
    spec: StaggeredDiDSpec,
    confidence_level: float = 0.95,
    figsize: tuple = (12, 8),
    title: str = "Event Study Plot (Callaway & Sant'Anna)",
    xlabel: str = "Event Time",
    ylabel: str = "ATT Estimate",
    reference_line_color: str = "gray",
    reference_line_style: str = "--",
    effect_color: str = "blue",
    confidence_color: str = "lightblue",
    marker: str = "o"
) -> Any:
    """
    Create an event study plot from Callaway & Sant'Anna estimation results.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
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
    if spec.model is None or not hasattr(spec.model, 'att_event'):
        raise ValueError("No Callaway & Sant'Anna results found in spec.model")
    
    # Extract results
    results = spec.model
    
    # Create dataframe for plotting
    event_data = results.summary[results.summary['type'] == 'event_time'].copy()
    
    # Ensure we have event time data
    if event_data.empty:
        raise ValueError("No event study estimates found in results")
    
    # Sort by event time
    event_data = event_data.sort_values('event_time')
    
    # Calculate confidence intervals
    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    event_data['lower_ci'] = event_data['att'] - z_value * event_data['se']
    event_data['upper_ci'] = event_data['att'] + z_value * event_data['se']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence intervals
    for _, row in event_data.iterrows():
        ax.plot(
            [row['event_time'], row['event_time']],
            [row['lower_ci'], row['upper_ci']],
            color=confidence_color,
            linewidth=2,
            alpha=0.7,
            zorder=1
        )
    
    # Plot point estimates
    ax.scatter(
        event_data['event_time'],
        event_data['att'],
        color=effect_color,
        marker=marker,
        s=50,
        zorder=2,
        label='ATT Estimate'
    )
    
    # Add zero line
    ax.axhline(y=0, color=reference_line_color, linestyle=reference_line_style, zorder=0)
    
    # Add vertical line at event time 0
    ax.axvline(x=0, color=reference_line_color, linestyle=reference_line_style, zorder=0)
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Add legend
    ax.legend()
    
    # Set x-ticks to integer values
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    return fig


# Helper function to summarize CS results
def format_cs_results(results: CSResults) -> str:
    """
    Format Callaway & Sant'Anna results as a readable string.
    
    Args:
        results: Callaway & Sant'Anna results object
        
    Returns:
        Formatted string with summary of results
    """
    import io
    buffer = io.StringIO()
    
    # Overall ATT
    buffer.write("==================================================\n")
    buffer.write("CALLAWAY & SANT'ANNA (2021) DiD RESULTS\n")
    buffer.write("==================================================\n\n")
    
    buffer.write(f"Overall ATT: {results.att_overall:.4f}\n")
    
    # Standard error and 95% CI for overall ATT
    se_overall = results.standard_errors.get('overall', np.nan)
    if not np.isnan(se_overall):
        buffer.write(f"Standard Error: {se_overall:.4f}\n")
        buffer.write(f"95% CI: [{results.att_overall - 1.96 * se_overall:.4f}, {results.att_overall + 1.96 * se_overall:.4f}]\n")
    
    buffer.write("\n")
    
    # ATT by cohort
    buffer.write("--------------------------------------------------\n")
    buffer.write("ATT by Cohort:\n")
    buffer.write("--------------------------------------------------\n")
    
    for cohort in sorted(results.att_cohort.keys()):
        att = results.att_cohort[cohort]
        se = results.standard_errors.get('cohort', {}).get(cohort, np.nan)
        buffer.write(f"Cohort {cohort}: {att:.4f}")
        if not np.isnan(se):
            buffer.write(f" (SE: {se:.4f})")
        buffer.write("\n")
    
    buffer.write("\n")
    
    # ATT by event time
    buffer.write("--------------------------------------------------\n")
    buffer.write("ATT by Event Time:\n")
    buffer.write("--------------------------------------------------\n")
    
    for event_time in sorted(results.att_event.keys()):
        att = results.att_event[event_time]
        se = results.standard_errors.get('event_time', {}).get(event_time, np.nan)
        buffer.write(f"Event time {event_time}: {att:.4f}")
        if not np.isnan(se):
            buffer.write(f" (SE: {se:.4f})")
        buffer.write("\n")
    
    # Return formatted string
    return buffer.getvalue()


@make_transformable
def format_callaway_santanna_results(spec: StaggeredDiDSpec) -> str:
    """
    Format Callaway & Sant'Anna results from a StaggeredDiDSpec.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        
    Returns:
        Formatted string with results
    """
    # Check if model exists
    if spec.model is None:
        return "No Callaway & Sant'Anna results found in spec.model"
    
    # Return formatted results
    return format_cs_results(spec.model) 