"""
PyAutoCausal Example Graph

This module demonstrates a comprehensive causal inference graph that automatically
selects appropriate methods based on data characteristics.
"""

from pathlib import Path
from typing import Optional
import webbrowser
import os

import pandas as pd
import numpy as np

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
from pyautocausal.pipelines.library.conditions import (
    has_multiple_periods, 
    has_staggered_treatment, 
    has_minimum_post_periods, 
    has_sufficient_never_treated_units, 
    has_single_treated_unit
)
from pyautocausal.pipelines.library.plots import event_study_plot, synthdid_plot
from pyautocausal.pipelines.library.callaway_santanna import (
    format_callaway_santanna_results, 
    event_study_plot_callaway
)

from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.orchestration.graph import ExecutableGraph


def _setup_output_directories(output_path: Path) -> tuple[Path, Path, Path]:
    """
    Create and return output subdirectories for plots, text, and notebooks.
    
    Args:
        output_path: Base output directory
        
    Returns:
        Tuple of (plots_dir, text_dir, notebooks_dir) as absolute paths
    """
    plots_dir = output_path / "plots"
    text_dir = output_path / "text"
    notebooks_dir = output_path / "notebooks"
    
    plots_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    notebooks_dir.mkdir(exist_ok=True)
    
    return plots_dir.absolute(), text_dir.absolute(), notebooks_dir.absolute()


def _create_basic_nodes(graph: ExecutableGraph, abs_text_dir: Path) -> None:
    """Create basic input and decision nodes for the graph."""
    # Input node for the dataframe
    graph.create_input_node("df", input_dtype=pd.DataFrame)

    # Decision nodes for routing
    graph.create_decision_node(
        'multi_period', 
        condition=has_multiple_periods.get_function(), 
        predecessors=["df"]
    )
    
    graph.create_decision_node(
        'single_treated_unit',
        condition=has_single_treated_unit.get_function(),
        predecessors=["multi_period"]
    )
    
    graph.create_decision_node(
        'multi_post_periods', 
        condition=has_minimum_post_periods.get_function(), 
        predecessors=["single_treated_unit"]
    )
    
    graph.create_decision_node(
        'stag_treat', 
        condition=has_staggered_treatment.get_function(), 
        predecessors=["multi_post_periods"]
    )


def _create_cross_sectional_branch(graph: ExecutableGraph, abs_text_dir: Path) -> None:
    """Create nodes for cross-sectional analysis branch."""
    # Cross-sectional specification and analysis
    graph.create_node(
        'stand_spec', 
        action_function=create_cross_sectional_specification.transform({'df': 'data'}), 
        predecessors=["multi_period"]
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


def _create_synthetic_did_branch(graph: ExecutableGraph, abs_plots_dir: Path) -> None:
    """Create nodes for Synthetic DiD analysis branch."""
    # Synthetic DiD specification and analysis
    graph.create_node(
        'synthdid_spec', 
        action_function=create_synthdid_specification.transform({'df': 'data'}), 
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


def _create_did_branch(graph: ExecutableGraph, abs_text_dir: Path) -> None:
    """Create nodes for standard DiD analysis branch."""
    # Standard DiD specification and analysis
    graph.create_node(
        'did_spec', 
        action_function=create_did_specification.transform({'df': 'data'}), 
        predecessors=["multi_period"]
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


def _create_event_study_branch(graph: ExecutableGraph, abs_text_dir: Path, abs_plots_dir: Path) -> None:
    """Create nodes for event study analysis branch."""
    # Event study specification and analysis
    graph.create_node(
        'event_spec', 
        action_function=create_event_study_specification.transform({'df': 'data'}), 
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


def _create_staggered_did_branch(graph: ExecutableGraph, abs_text_dir: Path, abs_plots_dir: Path) -> None:
    """Create nodes for staggered DiD analysis branch."""
    # Staggered DiD specification and analysis
    graph.create_node(
        'stag_spec', 
        action_function=create_staggered_did_specification.transform({'df': 'data'}), 
        predecessors=["stag_treat"]
    )
    
    graph.create_node(
        'ols_stag', 
        action_function=fit_panel_ols.transform({'stag_spec': 'spec'}),
        predecessors=["stag_spec"]
    )
    
    # Decision node for Callaway & Sant'Anna method selection
    graph.create_decision_node(
        'has_never_treated', 
        condition=lambda stag_spec: has_sufficient_never_treated_units(stag_spec.data), 
        predecessors=["stag_spec"]
    )
    
    # Callaway & Sant'Anna with never-treated control
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
    
    # Callaway & Sant'Anna with not-yet-treated control
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
    
    # Staggered DiD plots and output
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


def _configure_decision_paths(graph: ExecutableGraph) -> None:
    """Configure the decision routing for the graph."""
    # Multi-period routing
    graph.when_false("multi_period", "stand_spec")
    graph.when_true("multi_period", "single_treated_unit")
    graph.when_true("multi_period", "did_spec")

    # Single treated unit routing
    graph.when_true("single_treated_unit", "synthdid_spec")
    graph.when_false("single_treated_unit", "multi_post_periods")

    # Multi post periods routing
    graph.when_true("multi_post_periods", "stag_treat")

    # Staggered treatment routing
    graph.when_true("stag_treat", "stag_spec")
    graph.when_false("stag_treat", "event_spec")
    
    # Callaway & Sant'Anna method selection
    graph.when_true("has_never_treated", "cs_never_treated")
    graph.when_false("has_never_treated", "cs_not_yet_treated")


def causal_pipeline(output_path: Path) -> ExecutableGraph:
    """
    Create a comprehensive causal inference pipeline that automatically selects
    appropriate methods based on data characteristics.
    
    The pipeline supports:
    - Cross-sectional analysis (single period)
    - Synthetic DiD (single treated unit, multiple periods)
    - Standard DiD (multiple periods, insufficient for staggered)
    - Event study (multiple periods, non-staggered treatment)
    - Staggered DiD with Callaway & Sant'Anna methods
    
    Args:
        output_path: Directory where results will be saved
        
    Returns:
        Configured ExecutableGraph ready for execution
    """
    # Initialize graph and directories
    graph = ExecutableGraph(output_path=output_path)
    abs_plots_dir, abs_text_dir, abs_notebooks_dir = _setup_output_directories(output_path)
    
    # Create all nodes organized by analysis type
    _create_basic_nodes(graph, abs_text_dir)
    _create_cross_sectional_branch(graph, abs_text_dir)
    _create_synthetic_did_branch(graph, abs_plots_dir)
    _create_did_branch(graph, abs_text_dir)
    _create_event_study_branch(graph, abs_text_dir, abs_plots_dir)
    _create_staggered_did_branch(graph, abs_text_dir, abs_plots_dir)
    
    # Configure decision routing
    _configure_decision_paths(graph)
    
    return graph


def _print_execution_summary(graph: ExecutableGraph) -> None:
    """Print a summary of graph execution results."""
    executed_nodes = sum(
        1 for node in graph.nodes() 
        if hasattr(node, 'state') and node.state.name == 'COMPLETED'
    )
    skipped_nodes = sum(
        1 for node in graph.nodes() 
        if hasattr(node, 'state') and node.state.name == 'PASSED'
    )
    total_nodes = len(list(graph.nodes()))
    
    print(f"Total nodes in graph: {total_nodes}")
    print(f"Executed nodes: {executed_nodes}")
    print(f"Skipped nodes (due to branching): {skipped_nodes}")


def _export_outputs(graph: ExecutableGraph, output_path: Path) -> None:
    """Export graph visualization, notebook, and HTML report."""
    # Graph visualization
    md_visualization_path = output_path / "text" / "causal_pipeline_visualization.md"
    visualize_graph(graph, save_path=str(md_visualization_path))
    print(f"Graph visualization saved to {md_visualization_path}")
    
    # Notebook and HTML export
    notebook_path = output_path / "notebooks" / "causal_pipeline_execution.ipynb"
    html_path = output_path / "notebooks" / "causal_pipeline_execution.html"
    data_csv_path = output_path / "notebooks" / "causal_pipeline_data.csv"
    
    exporter = NotebookExporter(graph)
    
    # Export notebook
    exporter.export_notebook(
        str(notebook_path),
        data_path="causal_pipeline_data.csv",  # Relative path for notebook execution
        loading_function="pd.read_csv"
    )
    print(f"Notebook exported to {notebook_path}")
    
    # Export and run to HTML
    try:
        if data_csv_path.exists():
            # Use relative path for notebook execution
            html_output_path = exporter.export_and_run_to_html(
                notebook_filepath=notebook_path,
                html_filepath=html_path,
                data_path="causal_pipeline_data.csv",  # Relative path from notebooks directory
                loading_function="pd.read_csv",
                timeout=300  # 5 minutes timeout
            )
            print(f"HTML report with executed results exported to {html_output_path}")
        else:
            # Fallback: just convert existing notebook to HTML without execution
            from pyautocausal.persistence.notebook_runner import convert_notebook_to_html
            html_output_path = convert_notebook_to_html(notebook_path, html_path)
            print(f"HTML report (static) exported to {html_output_path}")
    
    except Exception as e:
        print(f"HTML export failed: {e}")
        print("Notebook is still available for manual inspection")


def main():
    """
    Main execution function for running the example graph.
    
    This demonstrates the full workflow:
    1. Generate mock data with staggered treatment
    2. Execute the causal inference graph
    3. Export results and visualizations
    """
    # Setup
    output_path = Path('output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("======= Running PyAutoCausal Example Graph =======")
    
    # Initialize graph
    graph = causal_pipeline(output_path)
    
    # Generate realistic mock data for demonstration
    data = generate_mock_data(
        n_units=100, 
        n_periods=10, 
        n_treated=70, 
        staggered_treatment=True, 
        noise_to_signal_ratio=1.5
    )
    
    # Save data for reference
    data_csv_path = output_path / "notebooks" / "causal_pipeline_data.csv"
    data.to_csv(data_csv_path, index=False)
    print(f"Mock data generated and saved to {data_csv_path}")

    # Display data characteristics
    all_units = data['id_unit'].unique()
    ever_treated_units = data[data['treat'] == 1]['id_unit'].unique()
    never_treated_units = set(all_units) - set(ever_treated_units)
    never_treated_ratio = len(never_treated_units) / len(all_units)
    
    print(f"Total units: {len(all_units)}, Ever treated: {len(ever_treated_units)}")
    print(f"Never-treated units: {never_treated_ratio * 100:.1f}%")
    print("This will trigger the Callaway & Sant'Anna branch with staggered treatment.\n")

    # Execute graph
    try:
        print("Executing causal inference graph...")
        graph.fit(df=data)
        print("Graph execution completed successfully.")
        
    except Exception as e:
        print(f"Error during graph execution: {e}")
        return

    # Results summary and export
    print("\n======= Execution Summary =======")
    _print_execution_summary(graph)
    print("-" * 50)
    
    _export_outputs(graph, output_path)
    print("\n======= Example Graph Run Finished =======")


if __name__ == "__main__":
    main()
