"""Analysis branches for different causal inference methods.

This module contains the specific implementations for each causal analysis branch:
- Cross-sectional analysis 
- Synthetic DiD
- Standard DiD
- Event study
- Staggered DiD with Callaway & Sant'Anna methods
"""

from pathlib import Path
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType

from pyautocausal.pipelines.library.estimators import (
    fit_ols, 
    fit_callaway_santanna_estimator, 
    fit_callaway_santanna_nyt_estimator, 
    fit_synthdid_estimator,
    fit_panel_ols, 
    fit_did_panel
)
from pyautocausal.pipelines.library.output import (
    write_statsmodels_summary, 
    write_statsmodels_summary_notebook
)
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, 
    create_did_specification, 
    create_event_study_specification, 
    create_staggered_did_specification, 
    create_synthdid_specification
)
from pyautocausal.pipelines.library.plots import event_study_plot, synthdid_plot
from pyautocausal.pipelines.library.callaway_santanna import (
    format_callaway_santanna_results, 
    event_study_plot_callaway
)


def create_cross_sectional_branch(graph: ExecutableGraph, abs_text_dir: Path) -> None:
    """Create nodes for cross-sectional analysis branch.
    
    This branch handles single-period data using OLS regression.
    """
    graph.create_node(
        'stand_spec', 
        action_function=create_cross_sectional_specification.transform({'cross_sectional_cleaned_data': 'data'}), 
        predecessors=["cross_sectional_cleaned_data"]
    )

    graph.create_node(
        'ols_stand', 
        action_function=fit_ols.transform({'stand_spec': 'spec'}),
        predecessors=["stand_spec"]
    )

    graph.create_node(
        'ols_stand_output',
        action_function=write_statsmodels_summary.transform({'ols_stand': 'res'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'ols_stand_output', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_stand"]
    )


def create_synthetic_did_branch(graph: ExecutableGraph, abs_plots_dir: Path) -> None:
    """Create nodes for Synthetic DiD analysis branch.
    
    This branch handles panel data with a single treated unit using synthetic controls.
    """
    graph.create_node(
        'synthdid_spec', 
        action_function=create_synthdid_specification.transform({'panel_cleaned_data': 'data'}), 
        predecessors=["single_treated_unit"]
    )

    graph.create_node(
        'synthdid_fit', 
        action_function=fit_synthdid_estimator.transform({'synthdid_spec': 'spec'}),
        predecessors=["synthdid_spec"]
    )
    
    graph.create_node(
        'synthdid_plot',
        action_function=synthdid_plot.transform({'synthdid_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_plots_dir / 'synthdid_plot', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["synthdid_fit"]
    )


def create_did_branch(graph: ExecutableGraph, abs_text_dir: Path) -> None:
    """Create nodes for standard DiD analysis branch.
    
    This branch handles panel data with insufficient periods for event studies,
    using simple difference-in-differences.
    """
    graph.create_node(
        'did_spec', 
        action_function=create_did_specification.transform({'multi_post_periods': 'data'}), 
        predecessors=["multi_post_periods"]
    )

    graph.create_node(
        'ols_did', 
        action_function=fit_did_panel.transform({'did_spec': 'spec'}),
        predecessors=["did_spec"]
    )
    
    graph.create_node(
        'save_ols_did',
        action_function=write_statsmodels_summary.transform({'ols_did': 'res'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'save_ols_did', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_did"]
    )
    
    graph.create_node(
        'save_did_output',
        action_function=write_statsmodels_summary.transform({'ols_did': 'res'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'save_did_output', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_did"]
    )


def create_event_study_branch(graph: ExecutableGraph, abs_text_dir: Path, abs_plots_dir: Path) -> None:
    """Create nodes for event study analysis branch.
    
    This branch handles panel data with sufficient periods for dynamic treatment effects,
    but without staggered treatment timing.
    """
    graph.create_node(
        'event_spec', 
        action_function=create_event_study_specification.transform({'panel_cleaned_data': 'data'}), 
        predecessors=["stag_treat"]
    )
    
    graph.create_node(
        'ols_event', 
        action_function=fit_panel_ols.transform({'event_spec': 'spec'}),
        predecessors=["event_spec"]
    )
    
    graph.create_node(
        'event_plot', 
        action_function=event_study_plot.transform({'ols_event': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_plots_dir / 'event_study_plot', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["ols_event"]
    )
    
    graph.create_node(
        'save_event_output',
        action_function=write_statsmodels_summary.transform({'ols_event': 'res'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'save_event_output', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_event"]
    )


def create_staggered_did_branch(graph: ExecutableGraph, abs_text_dir: Path, abs_plots_dir: Path) -> None:
    """Create nodes for staggered DiD analysis branch.
    
    This branch handles panel data with staggered treatment timing, using both
    traditional event studies and modern Callaway & Sant'Anna methods.
    """
    # Traditional staggered DiD specification and analysis
    graph.create_node(
        'stag_spec', 
        action_function=create_staggered_did_specification.transform({'panel_cleaned_data': 'data'}), 
        predecessors=["stag_treat"]
    )
    
    graph.create_node(
        'ols_stag', 
        action_function=fit_panel_ols.transform({'stag_spec': 'spec'}),
        predecessors=["stag_spec"]
    )
    
    # === CALLAWAY & SANT'ANNA METHOD SELECTION ===
    
    # Decision node for Callaway & Sant'Anna method selection
    from pyautocausal.pipelines.library.conditions import has_sufficient_never_treated_units
    graph.create_decision_node(
        'has_never_treated', 
        condition=lambda stag_spec: has_sufficient_never_treated_units(stag_spec.data), 
        predecessors=["stag_spec"]
    )
    
    # === CALLAWAY & SANT'ANNA METHODS ===
    
    # Callaway & Sant'Anna with never-treated control group
    graph.create_node(
        'cs_never_treated',
        action_function=fit_callaway_santanna_estimator.transform({'stag_spec': 'spec'}),
        predecessors=["has_never_treated"]
    )
    
    graph.create_node(
        'save_cs_never_treated',
        action_function=format_callaway_santanna_results.transform({'cs_never_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'callaway_santanna_never_treated_results',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["cs_never_treated"]
    )
    
    graph.create_node(
        'cs_never_treated_plot',
        action_function=event_study_plot_callaway.transform({'cs_never_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_plots_dir / 'callaway_santanna_never_treated_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["cs_never_treated"]
    )
    
    # Callaway & Sant'Anna with not-yet-treated control group
    graph.create_node(
        'cs_not_yet_treated',
        action_function=fit_callaway_santanna_nyt_estimator.transform({'stag_spec': 'spec'}),
        predecessors=["has_never_treated"]
    )
    
    graph.create_node(
        'save_cs_not_yet_treated',
        action_function=format_callaway_santanna_results.transform({'cs_not_yet_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'callaway_santanna_not_yet_treated_results',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["cs_not_yet_treated"]
    )
    
    graph.create_node(
        'cs_not_yet_treated_plot',
        action_function=event_study_plot_callaway.transform({'cs_not_yet_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_plots_dir / 'callaway_santanna_not_yet_treated_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["cs_not_yet_treated"]
    )
    
    # === TRADITIONAL STAGGERED DID OUTPUTS ===
    
    graph.create_node(
        'stag_event_plot',
        action_function=event_study_plot.transform({'ols_stag': 'spec'}),
        output_config=OutputConfig(
            output_filename=abs_plots_dir / 'staggered_event_study_plot', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["ols_stag"]
    )
    
    graph.create_node(
        'save_stag_output',
        action_function=write_statsmodels_summary.transform({'ols_stag': 'res'}),
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'save_stag_output', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_stag"]
    )


def create_all_analysis_branches(graph: ExecutableGraph, abs_text_dir: Path, abs_plots_dir: Path) -> None:
    """Create all analysis branches for the causal inference pipeline.
    
    This function adds all the specific causal analysis methods to the graph.
    The core decision structure (created separately) determines which branches are executed.
    
    Args:
        graph: The ExecutableGraph to add analysis nodes to
        abs_text_dir: Directory for text outputs
        abs_plots_dir: Directory for plot outputs
    """
    create_cross_sectional_branch(graph, abs_text_dir)
    create_synthetic_did_branch(graph, abs_plots_dir)
    create_did_branch(graph, abs_text_dir)
    create_event_study_branch(graph, abs_text_dir, abs_plots_dir)
    create_staggered_did_branch(graph, abs_text_dir, abs_plots_dir) 