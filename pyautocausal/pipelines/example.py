from pathlib import Path
from typing import Callable, Optional
import pandas as pd 
import statsmodels.api as sm
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.pipelines.library import DoubleMLNode, OLSNode
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.persistence.visualizer import visualize_graph

condition_nObs_DoubleML = lambda df: len(df) > 100

condition_nObs_OLS = lambda df: len(df) <= 100

def preprocess_lalonde_data() -> pd.DataFrame:
    """Load and preprocess the LaLonde dataset."""
    url = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
    df = pd.read_csv(url)
    y = df['re78']
    t = df['treat']
    X = df.drop(columns=['re78', 'treat','ID'])
    return pd.DataFrame({'y': y, 'treat': t, **X})

def create_causal_graph(output_path: Path):
    """Create the causal graph using GraphBuilder."""
    

    # Build graph using builder pattern
    graph = (ExecutableGraph(output_path=output_path)
        .create_input_node("df", input_dtype=pd.DataFrame)
        .create_decision_node(
            "doubleml_condition",
            condition_nObs_DoubleML,
            predecessors=["df"],
        )
        .create_node(
            "doubleml",
            DoubleMLNode.action,
            predecessors=["df"],
                output_filename="doubleml_results",
            output_config=OutputConfig(
                output_type=OutputType.TEXT 
            )
        )
        .create_node(
            "ols",
            OLSNode.action,
            predecessors=["df"],
            output_filename="ols_results",
            output_config=OutputConfig(
                output_type=OutputType.TEXT 
            )
        )
        .when_true("doubleml_condition", "doubleml")
        .when_false("doubleml_condition", "ols")
        .build())
    
    return graph

# For convenience, if someone runs this module directly
if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    
    # Create and execute graph
    graph = create_causal_graph(path)
    graph.fit(df=preprocess_lalonde_data())

    visualize_graph(graph, path / "graph.png")
