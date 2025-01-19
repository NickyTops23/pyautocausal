import pandas as pd
from pyautocausal.orchestration.nodes import Node, ExecutableGraph, RunContext
import pytest

def test_function_with_defaults(tmp_path):
    def process_data(df: pd.DataFrame, n_jobs: int = 1, verbose: bool = False):
        return f"Processed with {n_jobs} jobs, verbose={verbose}"
    
    graph = ExecutableGraph()
    
    # Create nodes
    data_node = Node("data", graph, lambda: pd.DataFrame())
    process_node = Node("process", graph, process_data)
    process_node.add_predecessor(data_node, argument_name="df")
    
    # Test with defaults
    graph.execute_graph()
    assert process_node.output == "Processed with 1 jobs, verbose=False"
    
    # Test with run context override
    graph = ExecutableGraph(run_context=RunContext())
    graph.run_context.n_jobs = 4
    
    data_node = Node("data", graph, lambda: pd.DataFrame())
    process_node = Node("process", graph, process_data)
    process_node.add_predecessor(data_node, argument_name="df")
    
    graph.execute_graph()
    assert process_node.output == "Processed with 4 jobs, verbose=False" 