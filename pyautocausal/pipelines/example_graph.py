from pathlib import Path
import pandas as pd
from pyautocausal.pipelines.library.estimators import fit_ols, fit_weighted_ols, fit_double_lasso
from pyautocausal.pipelines.library.output import write_statsmodels_summary, write_statsmodels_summary_notebook
from pyautocausal.pipelines.library.balancing import compute_synthetic_control_weights
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, create_did_specification, create_event_study_specification, create_staggered_did_specification, spec_constructor
)
from pyautocausal.pipelines.library.conditions import (
    has_multiple_periods, has_multiple_treated_units, has_staggered_treatment, has_never_treated_units, has_minimum_post_periods
)
from pyautocausal.pipelines.library.plots import event_study_plot

from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.nodes import Node, DecisionNode, InputNode
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from typing import Callable, Optional
import threading
import time


def simple_graph(output_path: Path):
    """Create a simplified causal graph without complex decision logic"""
    
    # Create the main executable graph
    graph = ExecutableGraph(output_path=output_path)
    
    # Create subdirectories if they don't exist
    plots_dir = output_path / "plots"
    text_dir = output_path / "text"
    notebooks_dir = output_path / "notebooks"
    
    plots_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    notebooks_dir.mkdir(exist_ok=True)
    
    # Calculate absolute paths to ensure correct file locations
    abs_plots_dir = plots_dir.absolute()
    abs_text_dir = text_dir.absolute()
    abs_notebooks_dir = notebooks_dir.absolute()
    
    # Add the input node for the dataframe
    
    graph.create_input_node("df", input_dtype=pd.DataFrame)

    # Create decision nodes
    graph.create_decision_node('multi_period', 
                                condition= has_multiple_periods.get_function(), 
                                predecessors=["df"])
    
    # First branch for standard specification using OLS
    # Create specification nodes
    graph.create_node('stand_spec', 
                    action_function=create_cross_sectional_specification.transform({'df': 'data'}), 
                    predecessors=["multi_period"])


    # Create OLS nodes with transformed parameter names - transform mapping: {'node_output': 'func_param'}
    graph.create_node('ols_stand', 
                     action_function=fit_ols.transform({'stand_spec': 'spec'}),
                     predecessors=["stand_spec"])


    # Add output formatting nodes with transformed parameter names
    graph.create_node('ols_stand_output',
                     action_function=write_statsmodels_summary.transform({'ols_stand': 'res'}),
                     output_config=OutputConfig(output_filename=abs_text_dir / 'ols_stand_output', output_type=OutputType.TEXT),
                     save_node=True,
                     predecessors=["ols_stand"])
    
    # Second branch for DiD and related methods
    graph.create_decision_node('multi_post_periods', 
                    condition=has_minimum_post_periods.get_function(), 
                    predecessors=["multi_period"])

    # Create DiD specification node
    graph.create_node('did_spec', 
                    action_function=create_did_specification.transform({'df': 'data'}), 
                    predecessors=["multi_period"])

    # Fit OLS on DiD specification
    graph.create_node('ols_did', 
                     action_function=fit_ols.transform({'did_spec': 'spec'}),
                     predecessors=["did_spec"])
    
    # Save DiD results
    graph.create_node('save_ols_did',
                    action_function=write_statsmodels_summary.transform({'ols_did': 'res'}),
                    output_config=OutputConfig(output_filename=abs_text_dir / 'save_ols_did', output_type=OutputType.TEXT),
                    save_node=True,
                    display_function=write_statsmodels_summary_notebook,
                    predecessors=["ols_did"])

    # Check for staggered treatment
    graph.create_decision_node('stag_treat', 
                    condition=has_staggered_treatment.get_function(), 
                    predecessors=["multi_post_periods"])

    # Create event study specification if no staggered treatment
    graph.create_node('event_spec', 
                    action_function=create_event_study_specification.transform({
                        'df': 'data'
                    }), 
                    predecessors=["stag_treat"])

    # Create staggered DiD specification if staggered treatment
    graph.create_node('stag_spec', 
                    action_function=create_staggered_did_specification.transform({'df': 'data'}), 
                    predecessors=["stag_treat"])
    
    # Fit OLS on Event Study specification
    graph.create_node('ols_event', 
                    action_function=fit_ols.transform({'stag_spec': 'spec'}),
                    predecessors=["event_spec","stag_spec"])
    
    # Create event study plot from event study results
    graph.create_node('event_plot', 
                    action_function=event_study_plot.transform({'ols_event': 'spec'}),
                    output_config=OutputConfig(output_filename=abs_plots_dir / 'event_study_plot', output_type=OutputType.PNG),
                    save_node=True,
                    predecessors=["ols_event"])
                    
    # Add debug output nodes for both models
    graph.create_node('save_did_output',
                     action_function=write_statsmodels_summary.transform({'ols_did': 'res'}),
                     output_config=OutputConfig(output_filename=abs_text_dir / 'save_did_output', output_type=OutputType.TEXT),
                     save_node=True,
                     predecessors=["ols_did"])
                     
    graph.create_node('save_event_output',
                     action_function=write_statsmodels_summary.transform({'ols_event': 'res'}),
                     output_config=OutputConfig(output_filename=abs_text_dir / 'save_event_output', output_type=OutputType.TEXT),
                     save_node=True,
                     predecessors=["ols_event"])

    # Configure decision paths
    graph.when_false("multi_period", "stand_spec")
    graph.when_true("multi_period", "multi_post_periods")
    graph.when_true("multi_period", "did_spec")

    graph.when_true("multi_post_periods", "stag_treat")

    graph.when_true("stag_treat", "stag_spec")
    graph.when_false("stag_treat", "event_spec")

    return graph  # Return the graph object

if __name__ == "__main__":
    # Run the simple example without the graph structure
    path = Path('output')
    path.mkdir(parents=True, exist_ok=True)
    graph = simple_graph(path)
    
    # Generate mock data with more units for proper synthetic control
    data = generate_mock_data(n_units=100, n_periods=10, n_treated=50, staggered_treatment=True, noise_to_signal_ratio=1.5)
    data.to_csv(path / "text" / "simple_graph_data.csv", index=False)
    try:
        graph.fit(df=data)
        
    except Exception as e:
        print(f"Error fitting graph: {e}")

    # Get all nodes that are instances of Node
    for node in graph.nodes():
        if hasattr(node, 'state'):
            name = getattr(node, 'name', 'Unknown')
            state = node.state.name if hasattr(node.state, 'name') else str(node.state)
            exec_count = getattr(node, 'execution_count', 'N/A')
            print(f"{name:<30} {state:<15} {exec_count}")
    print("-" * 50)

    visualize_graph(graph, save_path=str(path / "text" / "simple_graph.md"))
    print("\n\n")
    print("Exporting notebook...")
    notebook_path = path / "notebooks" / "simple_graph.ipynb"
    exporter = NotebookExporter(graph)
    exporter.export_notebook(notebook_path)
    print("Notebook exported")
    
    # Try to open the notebook in the default browser
    try:
        import webbrowser
        import os
        notebook_url = f"file://{os.path.abspath(notebook_path)}"
        print(f"Opening notebook at {notebook_url}")
        webbrowser.open(notebook_url)
    except Exception as e:
        print(f"Could not open notebook automatically: {e}")