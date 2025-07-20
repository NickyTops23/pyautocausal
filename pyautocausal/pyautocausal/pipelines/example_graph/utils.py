"""Utility functions for the causal inference pipeline.

This module contains helper functions for:
- Directory setup
- Execution summary and reporting  
- Output export (visualizations, notebooks, HTML)
"""

from pathlib import Path
from typing import Tuple
import webbrowser
import os
import pandas as pd

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.persistence.notebook_export import NotebookExporter


def setup_output_directories(output_path: Path) -> Tuple[Path, Path, Path]:
    """Create and return output subdirectories for plots, text, and notebooks.
    
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


def print_execution_summary(graph: ExecutableGraph) -> None:
    """Print a summary of graph execution results.
    
    Shows how many nodes were executed vs skipped due to decision branching.
    
    Args:
        graph: The ExecutableGraph that was executed
    """
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


def export_outputs(graph: ExecutableGraph, output_path: Path) -> None:
    """Export graph visualization, notebook, and HTML report.
    
    This function creates:
    1. A markdown visualization of the graph structure
    2. A Jupyter notebook with the analysis code
    3. An HTML report with executed results (if possible)
    
    Args:
        graph: The ExecutableGraph that was executed
        output_path: Base output directory
    """
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


def print_data_characteristics(data) -> None:
    """Print summary of data characteristics relevant to causal analysis.
    
    Args:
        data: The input DataFrame
    """
    all_units = data['id_unit'].unique()
    ever_treated_units = data[data['treat'] == 1]['id_unit'].unique()
    never_treated_units = set(all_units) - set(ever_treated_units)
    never_treated_ratio = len(never_treated_units) / len(all_units)
    
    print(f"Total units: {len(all_units)}, Ever treated: {len(ever_treated_units)}")
    print(f"Never-treated units: {never_treated_ratio * 100:.1f}%")
    
    # Determine expected analysis path
    periods = data['t'].nunique()
    if periods == 1:
        print("Single period data → Cross-sectional analysis")
    elif len(ever_treated_units) == 1:
        print("Single treated unit → Synthetic DiD analysis")
    elif data[data['treat'] == 1].groupby('id_unit')['t'].min().nunique() > 1:
        print("Staggered treatment → Callaway & Sant'Anna methods")
    else:
        print("Panel data → Standard DiD or Event study")


def create_simple_graph() -> ExecutableGraph:
    """Create a simple graph for testing purposes.
    
    This creates a basic pipeline with cross-sectional and DiD branches
    for testing serialization and basic functionality.
    
    Returns:
        Configured ExecutableGraph ready for execution
    """
    from pyautocausal.pipelines.library.conditions import has_multiple_periods
    from pyautocausal.pipelines.library.specifications import (
        create_cross_sectional_specification, 
        create_did_specification
    )
    from pyautocausal.pipelines.library.estimators import fit_ols, fit_did_panel
    from pyautocausal.pipelines.library.output import write_statsmodels_summary
    
    # Initialize graph
    graph = ExecutableGraph()
    
    # Create basic input node
    graph.create_input_node("df", input_dtype=pd.DataFrame)
    
    # Create multi-period decision node
    graph.create_decision_node(
        'multi_period', 
        condition=has_multiple_periods.get_function(), 
        predecessors=["df"]
    )
    
    # Cross-sectional branch
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
        save_node=True,
        predecessors=["ols_stand"]
    )
    
    # DiD branch
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
        save_node=True,
        predecessors=["ols_did"]
    )
    
    # Configure decision routing
    graph.when_false("multi_period", "stand_spec")
    graph.when_true("multi_period", "did_spec")
    
    return graph 