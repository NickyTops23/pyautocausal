from pathlib import Path
import pandas as pd
from pyautocausal.pipelines.library.estimators import fit_ols, fit_weighted_ols, fit_double_lasso, fit_callaway_santanna_estimator, fit_callaway_santanna_nyt_estimator, fit_synthdid_estimator
from pyautocausal.pipelines.library.output import write_statsmodels_summary, write_statsmodels_summary_notebook
from pyautocausal.pipelines.library.balancing import compute_synthetic_control_weights
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, create_did_specification, create_event_study_specification, create_staggered_did_specification, create_synthdid_specification, spec_constructor
)
from pyautocausal.pipelines.library.conditions import (
    has_multiple_periods, has_multiple_treated_units, has_staggered_treatment, 
    has_never_treated_units, has_minimum_post_periods, has_sufficient_never_treated_units, has_single_treated_unit
)
from pyautocausal.pipelines.library.plots import event_study_plot, synthdid_plot
from pyautocausal.pipelines.library.callaway_santanna import format_callaway_santanna_results, event_study_plot_callaway

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
    
    # Add decision node for single treated unit (for synthdid)
    graph.create_decision_node('single_treated_unit',
                                condition=has_single_treated_unit.get_function(),
                                predecessors=["multi_period"])
    
    # --------- Synthetic DiD Branch (single treated unit + multiple periods) ---------
    # Create synthdid specification node
    graph.create_node('synthdid_spec', 
                    action_function=create_synthdid_specification.transform({'df': 'data'}), 
                    predecessors=["single_treated_unit"])

    # Fit synthdid estimator
    graph.create_node('synthdid_fit', 
                     action_function=fit_synthdid_estimator.transform({'synthdid_spec': 'spec'}),
                     predecessors=["synthdid_spec"])
    
    # Create synthdid plot
    graph.create_node('synthdid_plot',
                     action_function=synthdid_plot.transform({'synthdid_fit': 'spec'}),
                     output_config=OutputConfig(output_filename=abs_plots_dir / 'synthdid_plot', output_type=OutputType.PNG),
                     save_node=True,
                     predecessors=["synthdid_fit"])
    # --------- End Synthetic DiD Branch ---------
    
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
                    predecessors=["single_treated_unit"])

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
                    action_function=fit_ols.transform({'event_spec': 'spec'}),
                    predecessors=["event_spec"])
    
    # Fit OLS on Staggered DiD specification
    graph.create_node('ols_stag', 
                     action_function=fit_ols.transform({'stag_spec': 'spec'}),
                     predecessors=["stag_spec"])
    
    # --------- Add Decision Node for Callaway & Sant'Anna ---------
    # Add a decision node to check if there are sufficient never-treated units
    # Use a direct lambda function that extracts data from stag_spec
    graph.create_decision_node('has_never_treated', 
                              condition=lambda stag_spec: has_sufficient_never_treated_units(stag_spec.data), 
                              predecessors=["stag_spec"])
    
    # --------- Add Callaway & Sant'Anna estimator with never-treated control ---------
    # Fit Callaway and Sant'Anna estimator on Staggered DiD specification using never-treated units
    graph.create_node('cs_never_treated',
                     action_function=fit_callaway_santanna_estimator.transform({'stag_spec': 'spec'}),
                     predecessors=["has_never_treated"])
    
    # Save Callaway and Sant'Anna (never-treated) results
    graph.create_node('save_cs_never_treated',
                     action_function=format_callaway_santanna_results.transform({'cs_never_treated': 'spec'}),
                     output_config=OutputConfig(
                         output_filename=abs_text_dir / 'callaway_santanna_never_treated_results',
                         output_type=OutputType.TEXT
                     ),
                     save_node=True,
                     display_function=display_cs_results_notebook,
                     predecessors=["cs_never_treated"])
    
    # Create Callaway and Sant'Anna (never-treated) event study plot
    graph.create_node('cs_never_treated_plot',
                     action_function=event_study_plot_callaway.transform({'cs_never_treated': 'spec'}),
                     output_config=OutputConfig(
                         output_filename=abs_plots_dir / 'callaway_santanna_never_treated_plot',
                         output_type=OutputType.PNG
                     ),
                     save_node=True,
                     predecessors=["cs_never_treated"])
    
    # --------- Add Callaway & Sant'Anna estimator with not-yet-treated control ---------
    # Fit Callaway and Sant'Anna estimator on Staggered DiD specification using not-yet-treated units
    graph.create_node('cs_not_yet_treated',
                     action_function=fit_callaway_santanna_nyt_estimator.transform({'stag_spec': 'spec'}),
                     predecessors=["has_never_treated"])
    
    # Save Callaway and Sant'Anna (not-yet-treated) results
    graph.create_node('save_cs_not_yet_treated',
                     action_function=format_callaway_santanna_results.transform({'cs_not_yet_treated': 'spec'}),
                     output_config=OutputConfig(
                         output_filename=abs_text_dir / 'callaway_santanna_not_yet_treated_results',
                         output_type=OutputType.TEXT
                     ),
                     save_node=True,
                     display_function=display_cs_results_notebook,
                     predecessors=["cs_not_yet_treated"])
    
    # Create Callaway and Sant'Anna (not-yet-treated) event study plot
    graph.create_node('cs_not_yet_treated_plot',
                     action_function=event_study_plot_callaway.transform({'cs_not_yet_treated': 'spec'}),
                     output_config=OutputConfig(
                         output_filename=abs_plots_dir / 'callaway_santanna_not_yet_treated_plot',
                         output_type=OutputType.PNG
                     ),
                     save_node=True,
                     predecessors=["cs_not_yet_treated"])
    # --------- End Callaway & Sant'Anna estimator ---------
    
    # Create event study plot from event study results
    graph.create_node('event_plot', 
                    action_function=event_study_plot.transform({'ols_event': 'spec'}),
                    output_config=OutputConfig(output_filename=abs_plots_dir / 'event_study_plot', output_type=OutputType.PNG),
                    save_node=True,
                    predecessors=["ols_event"])
    
    # Create event study plot from staggered DiD results
    graph.create_node('stag_event_plot',
                     action_function=event_study_plot.transform({'ols_stag': 'spec'}),
                     output_config=OutputConfig(output_filename=abs_plots_dir / 'staggered_event_study_plot', output_type=OutputType.PNG),
                     save_node=True,
                     predecessors=["ols_stag"])
                    
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
    
    graph.create_node('save_stag_output',
                     action_function=write_statsmodels_summary.transform({'ols_stag': 'res'}),
                     output_config=OutputConfig(output_filename=abs_text_dir / 'save_stag_output', output_type=OutputType.TEXT),
                     save_node=True,
                     predecessors=["ols_stag"])

    # Configure decision paths
    graph.when_false("multi_period", "stand_spec")
    graph.when_true("multi_period", "single_treated_unit")
    graph.when_true("multi_period", "did_spec")

    # Configure routing for single treated unit decision
    graph.when_true("single_treated_unit", "synthdid_spec")
    graph.when_false("single_treated_unit", "multi_post_periods")

    graph.when_true("multi_post_periods", "stag_treat")

    graph.when_true("stag_treat", "stag_spec")
    graph.when_false("stag_treat", "event_spec")
    
    # Configure new decision path for Callaway & Sant'Anna
    graph.when_true("has_never_treated", "cs_never_treated")  # If has never-treated units, use never-treated control
    graph.when_false("has_never_treated", "cs_not_yet_treated")  # Otherwise, use not-yet-treated control

    return graph  # Return the graph object

if __name__ == "__main__":
    # Run the simple example without the graph structure
    path = Path('output')
    path.mkdir(parents=True, exist_ok=True)
    
    print("======= Running PyAutoCausal Example Graph =======")
    graph = simple_graph(path)
    
    # Generate mock data 
    # Use multiple treated units with staggered treatment to trigger Callaway & Sant'Anna branch
    # This will test the Callaway & Sant'Anna functionality with never-treated controls
    data = generate_mock_data(n_units=100, n_periods=10, n_treated=70, staggered_treatment=True, noise_to_signal_ratio=1.5)
    data_csv_path = path / "notebooks" / "simple_graph_data.csv"
    data.to_csv(data_csv_path, index=False)
    print(f"Mock data generated and saved to {data_csv_path}")

    # Calculate the ratio of never-treated units for clarity
    all_units = data['id_unit'].unique()
    ever_treated_units = data[data['treat'] == 1]['id_unit'].unique()
    never_treated_units = set(all_units) - set(ever_treated_units)
    never_treated_ratio = len(never_treated_units) / len(all_units)
    
    print(f"Total units: {len(all_units)}, Treated units: {len(ever_treated_units)}")
    print(f"This will trigger the Callaway & Sant'Anna branch with staggered treatment.")
    print(f"Percentage of never-treated units: {never_treated_ratio * 100:.1f}%\\n")

    try:
        print("Fitting the graph...")
        graph.fit(df=data)
        print("Graph fitting complete.")
        
    except Exception as e:
        print(f"Error fitting graph: {e}")
        # Optionally, re-raise the exception if you want the script to stop on error
        # raise
    
    print("\n======= Example Graph Execution Summary =======")
    # Provide a more concise summary or simply state completion
    executed_nodes_count = sum(1 for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED')
    skipped_nodes_count = sum(1 for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'PASSED') # PASSED means skipped due to branching
    total_nodes = len(list(graph.nodes()))
    print(f"Total nodes in graph: {total_nodes}")
    print(f"Executed nodes: {executed_nodes_count}")
    print(f"Skipped nodes (due to branching): {skipped_nodes_count}")

    print("-" * 50)

    # Graph visualization and notebook export remain
    md_visualization_path = path / "text" / "simple_graph_visualization.md"
    visualize_graph(graph, save_path=str(md_visualization_path))
    print(f"Graph visualization saved as markdown to {md_visualization_path}")
    
    print("\nExporting notebook...")
    notebook_path = path / "notebooks" / "simple_graph_execution.ipynb"
    exporter = NotebookExporter(graph)
    exporter.export_notebook(notebook_path)
    print(f"Notebook exported to {notebook_path}")
    
    # Try to open the notebook in the default browser
    try:
        import webbrowser
        import os
        notebook_url = f"file://{os.path.abspath(notebook_path)}"
        print(f"Attempting to open notebook at {notebook_url}")
        # webbrowser.open(notebook_url) # Keep this commented out for non-interactive environments
    except Exception as e:
        print(f"Could not open notebook automatically: {e}")
    
    print("\n======= Example Graph Run Finished =======")