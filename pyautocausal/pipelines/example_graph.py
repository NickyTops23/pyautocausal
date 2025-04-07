from pathlib import Path
import pandas as pd
from pyautocausal.pipelines.library.estimators import fit_ols, fit_weighted_ols, fit_double_lasso
from pyautocausal.pipelines.library.output import write_statsmodels_summary
from pyautocausal.pipelines.library.balancing import compute_synethetic_control_weights
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, create_did_specification
)
from pyautocausal.pipelines.library.conditions import (
    has_multiple_periods, has_multiple_treated_units,
)
from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.nodes import Node, DecisionNode, InputNode
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


def simple_graph(output_path: Path):
    """Create a simplified causal graph without complex decision logic"""
    
    # Create the main executable graph
    graph = ExecutableGraph(output_path=output_path)
    
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
                     action_function=fit_ols.transform({'stand_spec': 'inputs'}),
                     predecessors=["stand_spec"])


    # Add output formatting nodes with transformed parameter names
    graph.create_node('ols_stand_output',
                     action_function=write_statsmodels_summary.transform({'ols_stand': 'res'}),
                     output_config=OutputConfig(output_filename='ols_stand_output', output_type=OutputType.TEXT),
                     save_node=True,
                     predecessors=["ols_stand"])
    
    # Second branch for DiD specification using OLS and potentially synthetic control
    graph.create_node('did_spec', 
                    action_function=create_did_specification.transform({'df': 'data'}),           
                    predecessors=["multi_period"])


    # Check whether to use synthetic control for if there is only one treated unit
    graph.create_decision_node('multi_treated_units', 
                     condition= lambda x: has_multiple_treated_units.get_function()(x.data),
                     predecessors=["did_spec"])
    # If True, use OLS
    graph.create_node('ols_did', 
                     action_function=fit_ols.transform({'did_spec': 'inputs'}),
                     predecessors=["multi_treated_units"])
    
    graph.create_node('ols_did_output',
                    action_function=write_statsmodels_summary.transform({'ols_did': 'res'}),
                    output_config=OutputConfig(output_filename='ols_did_output', output_type=OutputType.TEXT),
                    save_node=True,
                    predecessors=["ols_did"])
    
    # If False, use synthetic control
    graph.create_node('synth_control', 
                     action_function=compute_synethetic_control_weights.get_function(),
                     predecessors=["multi_treated_units"])

    graph.create_node('wols_did_synth', 
                     action_function=fit_weighted_ols.transform({'synth_control': 'inputs'}),
                     predecessors=["synth_control"])

    graph.create_node('wols_did_synth_output',
                     action_function=write_statsmodels_summary.transform({'wols_did_synth': 'res'}),
                     output_config=OutputConfig(output_filename='wols_did_synth_output', output_type=OutputType.TEXT),
                     save_node=True,
                     predecessors=["wols_did_synth"])
    
    # Configure decision paths
    graph.when_false("multi_period", "stand_spec")
    graph.when_true("multi_period", "did_spec")
    graph.when_false("multi_treated_units", "synth_control")
    graph.when_true("multi_treated_units", "ols_did")
    
    return graph  # Return the graph object

if __name__ == "__main__":
    # Run the simple example without the graph structure
    path = Path('output')
    path.mkdir(parents=True, exist_ok=True)
    graph = simple_graph(path)
    
    # Generate mock data with more units for proper synthetic control
    data = generate_mock_data(n_units=5, n_periods=6, n_treated=1)
    
    graph.fit(df=data)

    visualize_graph(graph, save_path=str(path / "simple_graph.png"))
    exporter = NotebookExporter(graph)
    exporter.export_notebook(path / "simple_graph.ipynb")