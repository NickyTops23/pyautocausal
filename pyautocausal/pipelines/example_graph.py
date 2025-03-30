from pathlib import Path
import pandas as pd
from pyautocausal.pipelines.library.models import OLSNode, PSMNode, DiDNode
from pyautocausal.pipelines.library.conditions import (
    TimeConditions, TreatmentConditions, PeriodConditions
)
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.persistence.output_config import OutputConfig, OutputType
"""
Heuristics for Selecting Causal Inference Methods:

- **One Period:**
  - Use OLS if groups are balanced.
  - For unbalanced samples, apply balancing methods before estimation.

- **Multiple Periods:**
  - If there is a single treated unit, use Synthetic Control.
  - With multiple treated units, use Difference-in-Differences (DiD).

- **Multiple Pre-treatment Periods:**
  - Check for parallel trends to validate the DiD approach.

- **Multiple Pre and Post-treatment Periods:**
  - Run an Event Study to analyze the dynamics of treatment effects.

- **Staggered Treatment Adoption:**
  - Adjust using advanced DiD techniques (e.g., based on Hautefeuille's work) to account for varying timings.


Things to add:

No pure control group?
PSM Matching for DiD

"""



def create_causal_graph(output_path: Path):
    """Create the causal graph using GraphBuilder."""
    
    # First create graph with condition nodes
    graph = (GraphBuilder(output_path=output_path)
                .add_input_node("df")
                
                # Single period branch
                .create_node(
                    name="single_period_true", 
                    action_function=lambda df: df,
                    condition=TimeConditions.multiple_periods(expected=False),
                    predecessors={'df': 'df'}
                )
                
                # Multiple periods branch
                .create_node(
                    name="single_period_false", 
                    action_function=lambda df: df,
                    condition=TimeConditions.multiple_periods(expected=True),
                    predecessors={'df': 'df'}
                )
                
                # Check number of treated units
                .create_node(
                    name="single_treated_true", 
                    action_function=lambda df: df,
                    condition=TreatmentConditions.multiple_treated_units(expected=False),
                    predecessors={'df': 'single_period_false'}
                )
                .create_node(
                    name="single_treated_false", 
                    action_function=lambda df: df,
                    condition=TreatmentConditions.multiple_treated_units(expected=True),
                    predecessors={'df': 'single_period_false'}
                )          
                # Check if there are sufficient pre-periods
                .create_node(
                    name="multiple_pre_periods_true", 
                    action_function=lambda df: df,
                    condition=PeriodConditions.sufficient_pre_periods(expected=True, min_periods=2),
                    predecessors={'df': 'single_treated_false'}
                )
                
                # Check if there are sufficient post-periods
                .create_node(
                    name="multiple_post_periods_true", 
                    action_function=lambda df: df,
                    condition=PeriodConditions.sufficient_post_periods(expected=True, min_periods=2),
                    predecessors={'df': 'multiple_pre_periods_true'}
                )
    )
    
    # Single Period Models
    graph = (graph
                .create_node(
                    name="naive_ols",
                    action_function=OLSNode.action,
                    predecessors={'df': 'single_period_true'}
                )
                .create_node(
                    name="output_naive_ols",
                    action_function=OLSNode.output,
                    predecessors={'model': 'naive_ols'},
                    output_config=OutputConfig(
                        output_filename="naive_ols_output",
                        output_type=OutputType.TEXT
                    ),
                    save_node=True
                )
    )

    graph = (graph
                .create_node(
                    name = 'psm_matching',
                    action_function=lambda df: PSMNode.action(df, n_neighbors=PSMNode.n_neighbors, caliper=PSMNode.caliper),
                    predecessors={'df': 'multiple_post_periods_true'}
                )
                .create_node(
                    name = 'output_psm_matching',
                    action_function=PSMNode.output,
                    predecessors={'psm_output': 'psm_matching'},
                    output_config=OutputConfig(
                        output_filename="psm_matching_output",
                        output_type=OutputType.TEXT
                    ),
                    save_node=True
                )
    )
    
    graph = (graph
                .create_node(
                    name="run_ols",
                    action_function=OLSNode.action,
                    condition=TreatmentConditions.imbalanced_covariates(expected=False),
                    predecessors={'df': 'multiple_post_periods_true'}
                )
                .create_node(
                    name = 'output_run_ols',
                    action_function=OLSNode.output,
                    predecessors={'model': 'run_ols'},
                    output_config=OutputConfig(
                        output_filename="run_ols_output",   
                        output_type=OutputType.TEXT
                    ),
                    save_node=True
                )
    )

    # DiD Implementation
    graph = (graph
                .create_node(
                    name="run_did",
                    action_function=DiDNode.action,
                    predecessors={'df': 'multiple_post_periods_true'}
                )
                .create_node(
                    name = 'output_run_did',
                    action_function=DiDNode.output,
                    predecessors={'model': 'run_did'},
                    output_config=OutputConfig(
                        output_filename="run_did_output",
                        output_type=OutputType.TEXT
                    ),
                    save_node=True
                )
    )
    
    return graph


if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    
    # Create graph
    graph = create_causal_graph(path)
    
    # Build and visualize the graph
    graph = graph.build()  # Need to call build() to finalize the graph

    data = generate_mock_data(
        n_units=200,
        n_periods=4,
        n_treated=100,
        treatment_effect=2.0,
        random_seed=42
    )
    
    data.to_csv("data.csv", index=False)
    # Fit graph
    graph.fit(df= data)

    # Visualize graph
    visualize_graph(graph, path / "causal_graph.png")
    exporter = NotebookExporter(graph)
    exporter.export_notebook(str(path / "causal_graph.ipynb"))