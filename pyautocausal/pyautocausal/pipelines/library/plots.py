"""
Plot functions for causal inference visualizations
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
import re
import warnings
import scipy.stats as stats

from pyautocausal.pipelines.library.specifications import (
    BaseSpec, DiDSpec, StaggeredDiDSpec, EventStudySpec
)
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.callaway_santanna import CSResults
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.pipelines.library.synthdid.plot import plot_synthdid

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


def _extract_event_study_results(spec: Union[EventStudySpec, StaggeredDiDSpec], params: pd.Series, conf_int: pd.DataFrame) -> pd.DataFrame:
    """Extract event study results from EventStudySpec or StaggeredDiDSpec."""
    periods = []
    coeffs = []
    lower_ci = []
    upper_ci = []
    
    event_pattern_pre = re.compile(r'event_pre(\d+)')   # For negative periods: event_pre1 = -1
    event_pattern_post = re.compile(r'event_post(\d+)') # For positive periods: event_post1 = +1
    
    # Determine which parameters to check
    if isinstance(spec, EventStudySpec):
        # For EventStudySpec, use the specific event columns
        param_names = sorted(spec.event_cols)
    else:
        # For StaggeredDiDSpec, check all parameters
        param_names = sorted(params.index)
    
    # Process parameters to find event study coefficients
    for param_name in param_names:
        if param_name not in params.index:
            continue
            
        # Check for pre-treatment period (pre prefix)
        match_pre = event_pattern_pre.match(param_name)
        if match_pre:
            period = -int(match_pre.group(1))  # Convert to negative number
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
            continue
            
        # Check for post-treatment period (post prefix)
        match_post = event_pattern_post.match(param_name)
        if match_post:
            period = int(match_post.group(1))  # Already positive
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
            continue
        
        # Check for event_period0 (t=0)
        if param_name == 'event_period0':
            periods.append(0)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    # Handle reference period
    if isinstance(spec, EventStudySpec):
        # For EventStudySpec, use the specified reference period
        if spec.reference_period not in periods:
            periods.append(spec.reference_period)
            coeffs.append(0)
            lower_ci.append(0)
            upper_ci.append(0)
    else:
        # For StaggeredDiDSpec, add treatment period (time 0) as reference if not present
        if 0 not in periods:
            periods.append(0)
            coeffs.append(0.0)
            lower_ci.append(0.0)
            upper_ci.append(0.0)
    
    return pd.DataFrame({
        'period': periods,
        'coef': coeffs,
        'lower': lower_ci,
        'upper': upper_ci
    })


def _extract_generic_results(params: pd.Series, conf_int: pd.DataFrame) -> pd.DataFrame:
    """Extract results from generic model with event patterns or treat_post fallback."""
    periods = []
    coeffs = []
    lower_ci = []
    upper_ci = []
    
    # Look for event_pre* columns (pre-treatment periods)
    event_pattern_pre = re.compile(r'event_pre(\d+)')
    for param_name in params.index:
        match = event_pattern_pre.match(param_name)
        if match:
            period = -int(match.group(1))  # Negative for pre-periods
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    # Look for event_post* columns (post-treatment periods)
    event_pattern_post = re.compile(r'event_post(\d+)')
    for param_name in params.index:
        match = event_pattern_post.match(param_name)
        if match:
            period = int(match.group(1))  # Positive for post-periods
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    # Check for event_period0 (t=0)
    if 'event_period0' in params.index:
        periods.append(0)
        coeffs.append(params['event_period0'])
        lower_ci.append(conf_int.loc['event_period0', 'lower'])
        upper_ci.append(conf_int.loc['event_period0', 'upper'])
    
    # Fallback to old event_* pattern for backward compatibility
    if not periods:
        event_pattern_old = re.compile(r'event_(-?\d+)')
        for param_name in params.index:
            match = event_pattern_old.match(param_name)
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
    
    # Add reference period (time 0) if not already present
    if 0 not in periods:
        periods.append(0)
        coeffs.append(0.0)
        lower_ci.append(0.0)
        upper_ci.append(0.0)
    
    return pd.DataFrame({
        'period': periods,
        'coef': coeffs,
        'lower': lower_ci,
        'upper': upper_ci
    })


@make_transformable
def event_study_plot(spec: Union[StaggeredDiDSpec, EventStudySpec], 
                    confidence_level: float = 0.95,
                    figsize: tuple = (12, 8),
                    title: str = "Event Study Plot",
                    xlabel: str = "Event Time",
                    ylabel: str = "Coefficient Estimate",
                    reference_line_color: str = "gray",
                    reference_line_style: str = "--",
                    effect_color: str = "blue",
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
        marker: Marker style for point estimates
        
    Returns:
        Matplotlib figure with the event study plot
    """
    # Check if model exists
    if spec.model is None:
        raise ValueError("Specification must contain a fitted model")

    # For standard OLS models
    params = spec.model.params
    conf_int = _get_confidence_intervals(spec.model, confidence_level)
    
    # Extract results from the specification
    results_df = _extract_event_study_results(spec, params, conf_int)

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


@make_transformable
def callaway_santanna_summary_table(spec: StaggeredDiDSpec) -> pd.DataFrame:
    """
    Create a comprehensive summary table for Callaway & Sant'Anna results.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model using csdid
        
    Returns:
        DataFrame with formatted results table
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    
    # Get the summary table from the att_gt object
    summary_table = att_gt.summary2.copy()
    
    # Round numerical columns for better presentation
    numerical_cols = summary_table.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        summary_table[col] = summary_table[col].round(4)
    
    return summary_table


@make_transformable
def callaway_santanna_event_study_plot(
    spec: StaggeredDiDSpec,
    confidence_level: float = 0.95,
    figsize: tuple = (12, 8),
    title: str = "Event Study Plot (Callaway & Sant'Anna - csdid)",
    xlabel: str = "Event Time",
    ylabel: str = "ATT Estimate",
    reference_line_color: str = "gray",
    reference_line_style: str = "--",
    effect_color: str = "blue",
    marker: str = "o",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create an enhanced event study plot using the csdid module's plotting functionality.
    
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
        marker: Marker style for point estimates
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure with the event study plot
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    dynamic_effects = spec.model.get('dynamic_effects')
    
    if dynamic_effects is None:
        raise ValueError("No dynamic effects found. Run aggte with type='dynamic' first.")
    
    # Close any existing figures to avoid multiple windows
    plt.close('all')
    
    # Use the ATTgt plotting method - it will create its own figure
    fig = att_gt.plot_aggte(
        title=title,
        xlab=xlabel,
        ylab=ylabel,
        theming=True,
        ref_line=0
    )
    
    # Set the figure size
    fig.set_size_inches(figsize)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
    


@make_transformable
def callaway_santanna_group_plot(
    spec: StaggeredDiDSpec,
    figsize: tuple = (15, 10),
    title: str = "Group-Time Treatment Effects (Callaway & Sant'Anna)",
    xlabel: str = "Time Period",
    ylabel: str = "ATT Estimate",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create group-specific treatment effect plots using the csdid module.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        figsize: Figure size as (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure with group-specific plots
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    
    # Close any existing figures to avoid multiple windows
    plt.close('all')
    
    # Use the ATTgt plotting method - it will create its own figure
    fig = att_gt.plot_attgt(
        title=title,
        xlab=xlabel,
        ylab=ylabel,
        theming=True,
        ref_line=0,
        legend=True
    )
    
    # Set the figure size
    fig.set_size_inches(figsize)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


@make_transformable
def callaway_santanna_comprehensive_results(spec: StaggeredDiDSpec) -> str:
    """
    Generate simplified results summary for Callaway & Sant'Anna estimation.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        
    Returns:
        Formatted string with key results
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    overall_effect = spec.model.get('overall_effect')
    control_group = spec.model.get('control_group', 'unknown')
    
    import io
    buffer = io.StringIO()
    
    # Header
    buffer.write("=" * 60 + "\n")
    buffer.write("CALLAWAY & SANT'ANNA DiD RESULTS\n")
    buffer.write("=" * 60 + "\n")
    buffer.write(f"Control Group: {control_group.replace('_', ' ').title()}\n")
    buffer.write(f"Sample: {len(spec.data)} obs, {spec.data[spec.unit_col].nunique()} units\n\n")
    
    # Overall ATT - simplified and safer
    if overall_effect is not None:
        try:
            overall_att = float(overall_effect.atte['overall_att'])
            overall_se = float(overall_effect.atte['overall_se'])
            
            buffer.write("OVERALL TREATMENT EFFECT:\n")
            buffer.write("-" * 30 + "\n")
            buffer.write(f"ATT:        {overall_att:>8.4f}\n")
            buffer.write(f"Std Error:  {overall_se:>8.4f}\n")
            
            # Calculate confidence intervals
            ci_lower = overall_att - 1.96 * overall_se
            ci_upper = overall_att + 1.96 * overall_se
            buffer.write(f"95% CI:     [{ci_lower:>7.4f}, {ci_upper:>7.4f}]\n")
            
            # Simple significance test
            if overall_se > 0:
                t_stat = overall_att / overall_se
                significance = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
                buffer.write(f"Significant: {significance if significance else 'No'}\n")
            
            buffer.write("\n")
        except Exception as e:
            buffer.write(f"Overall effect formatting error: {str(e)}\n\n")
    
    # Simple summary table - just use the built-in summary
    buffer.write("DETAILED RESULTS:\n")
    buffer.write("-" * 30 + "\n")
    try:
        summary_table = att_gt.summary2
        buffer.write(summary_table.to_string(index=False))
        buffer.write("\n\n")
    except Exception as e:
        buffer.write(f"Summary table error: {str(e)}\n\n")
    
    buffer.write("=" * 60 + "\n")
    buffer.write("Note: *** p<0.01, ** p<0.05, * p<0.1\n")
    buffer.write("=" * 60 + "\n")
    
    return buffer.getvalue()


@make_transformable  
def callaway_santanna_diagnostic_plots(
    spec: StaggeredDiDSpec,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive diagnostic plots for Callaway & Sant'Anna estimation.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure with multiple diagnostic subplots
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    overall_effect = spec.model.get('overall_effect')
    dynamic_effects = spec.model.get('dynamic_effects')
    group_effects = spec.model.get('group_effects')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Callaway & Sant\'Anna DiD - Comprehensive Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Event Study (Dynamic Effects)
    if dynamic_effects is not None:
        ax1 = axes[0, 0]
        try:
            dynamic_results = dynamic_effects.atte
            event_times = [int(float(x)) for x in dynamic_results['egt']]
            atts = [float(x) for x in dynamic_results['att_egt']]
            ses = [float(x) for x in dynamic_results['se_egt']]
            
            # Plot with error bars
            ax1.errorbar(event_times, atts, yerr=1.96*np.array(ses), 
                        fmt='o', color='blue', capsize=5, capthick=2)
            ax1.axhline(y=0, color='gray', linestyle='--')
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Event Time')
            ax1.set_ylabel('ATT')
            ax1.set_title('Dynamic Treatment Effects')
            ax1.grid(True, alpha=0.3)
        except Exception as e:
            ax1.text(0.5, 0.5, f"Dynamic effects plot error:\n{str(e)}", 
                    transform=ax1.transAxes, ha='center', va='center')
    
    # Plot 2: Group-specific Effects
    if group_effects is not None:
        ax2 = axes[0, 1]
        try:
            group_results = group_effects.atte
            groups = [int(float(x)) for x in group_results['egt']]
            atts = [float(x) for x in group_results['att_egt']]
            ses = [float(x) for x in group_results['se_egt']]
            
            # Plot with error bars
            ax2.errorbar(groups, atts, yerr=1.96*np.array(ses), 
                        fmt='s', color='red', capsize=5, capthick=2)
            ax2.axhline(y=0, color='gray', linestyle='--')
            ax2.set_xlabel('Treatment Group')
            ax2.set_ylabel('ATT')
            ax2.set_title('Group-Specific Effects')
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(0.5, 0.5, f"Group effects plot error:\n{str(e)}", 
                    transform=ax2.transAxes, ha='center', va='center')
    
    # Plot 3: Treatment Effect Distribution
    ax3 = axes[1, 0]
    results = att_gt.results
    all_atts = [att for att in results['att'] if not np.isnan(att)]
    if all_atts:
        ax3.hist(all_atts, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=np.mean(all_atts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_atts):.3f}')
        ax3.set_xlabel('ATT Estimates')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Group-Time ATT Estimates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics Table as Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = []
    if overall_effect is not None:
        overall_att = overall_effect.atte['overall_att']
        overall_se = overall_effect.atte['overall_se']
        summary_text.append(f"Overall ATT: {float(overall_att):.4f} ({float(overall_se):.4f})")
    
    if group_effects is not None:
        n_groups = len(group_effects.atte['egt'])
        summary_text.append(f"Number of Groups: {n_groups}")
    
    if dynamic_effects is not None:
        n_periods = len(dynamic_effects.atte['egt'])
        summary_text.append(f"Event Time Periods: {n_periods}")
    
    # Add data characteristics
    data = spec.data
    n_units = data[spec.unit_col].nunique()
    n_periods = data[spec.time_col].nunique()
    ever_treated = data[data['treat'] == 1][spec.unit_col].nunique()
    
    summary_text.extend([
        f"Total Units: {n_units}",
        f"Time Periods: {n_periods}",
        f"Ever Treated Units: {ever_treated}",
        f"Control Group: {spec.model.get('control_group', 'Unknown').title()}"
    ])
    
    # Display summary text
    text_str = '\n'.join(summary_text)
    ax4.text(0.1, 0.9, 'Summary Statistics:', fontsize=14, fontweight='bold', 
            transform=ax4.transAxes, verticalalignment='top')
    ax4.text(0.1, 0.8, text_str, fontsize=11, transform=ax4.transAxes, 
            verticalalignment='top', family='monospace')
    
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig

@make_transformable
def hainmueller_synth_plot(spec, output_config: Optional[OutputConfig] = None, 
                          figsize: tuple = (12, 8), **kwargs) -> plt.Figure:
    """
    Create a plot for Hainmueeller synthetic control results with placebo tests.
    
    Args:
        spec: A SynthDIDSpec object with fitted Hainmueeller model
        output_config: Configuration for saving the plot
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments for plot customization
        
    Returns:
        matplotlib figure object
    """
    model = spec.hainmueller_model

    treated_unit = spec.data[spec.data[spec.treatment_cols[0]] == 1][spec.unit_col].unique()[0]

    model.plot(["original", "pointwise", "cumulative", "in-space placebo", "rmspe ratio"], treated_label=treated_unit, synth_label=f"Synthetic {treated_unit}")

