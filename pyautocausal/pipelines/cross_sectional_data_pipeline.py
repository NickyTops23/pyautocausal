from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.pipelines.library import DoubleMLNode, OLSNode
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.orchestration.condition import create_condition
from typing import Dict


def condition_nObs_DoubleML(df: pd.DataFrame) -> bool:
    return len(df) > 100

def condition_nObs_OLS(df: pd.DataFrame) -> bool:
    return len(df) <= 100

def create_causal_graph(output_path: Path):
    """Create the causal graph using GraphBuilder."""
    
    # Define reusable conditions
    doubleml_condition = create_condition(
        condition_nObs_DoubleML,
        "Sample size is greater than 100 observations"
    )

    ols_condition = create_condition(
        condition_nObs_OLS,
        "Sample size is less than or equal to 100 observations"
    )

    # Build graph using builder pattern
    graph = (GraphBuilder(output_path=output_path)
        .add_input_node("df")
        .create_node(
            "doubleml",
            DoubleMLNode.action,
            predecessors={"df": "df"},
            condition=doubleml_condition,
            save_node=True,
            output_config=OutputConfig(
                output_filename="doubleml_results",
                output_type=OutputType.TEXT
            )
        )
        .create_node(
            "ols",
            OLSNode.action,
            predecessors={"df": "df"},
            condition=ols_condition,
            save_node=True,
            output_config=OutputConfig(
                output_filename="ols_results",
                output_type=OutputType.TEXT
            )
        )
        
        .build())
    
    return graph

def get_output_nodes(self) -> Dict[str, Node]:
    """Returns a dictionary of all terminal nodes (nodes without successors)"""
    output_nodes = {}
    for node in self.nodes():
        if isinstance(node, Node) and not list(self.successors(node)):
            output_nodes[node.name] = node
    return output_nodes

# For convenience, if someone runs this module directly
if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    
    # Create and execute graph
    graph = create_causal_graph(path)
    graph.fit(df=preprocess_lalonde_data())

class Node:
    def __rshift__(self, other: 'Node') -> None:
        """Implements the -> operator for node wiring"""
        if not isinstance(other, InputNode):
            raise ValueError("Right-hand node must be an input node")
        self.add_successor(other)
        return self