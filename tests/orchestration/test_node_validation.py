import pytest
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.nodes import Node, InputNode

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def test_duplicate_node_name():
    """Test that adding a node with a duplicate name raises an error"""
    graph = ExecutableGraph()
    
    # Add first node
    node1 = Node(name="process", action_function=process_dataframe)
    graph.add_node_to_graph(node1)
    
    # Try to add second node with same name
    node2 = Node(name="process", action_function=process_dataframe)
    with pytest.raises(ValueError) as exc_info:
        graph.add_node_to_graph(node2)
    assert "already exists in the graph" in str(exc_info.value)

def test_duplicate_input_node_name():
    """Test that adding an input node with a duplicate name raises an error"""
    graph = ExecutableGraph()
    
    # Add first input node
    input1 = InputNode(name="input", input_dtype=pd.DataFrame)
    graph.add_node_to_graph(input1)
    
    # Try to add second input node with same name
    input2 = InputNode(name="input", input_dtype=pd.DataFrame)
    with pytest.raises(ValueError) as exc_info:
        graph.add_node_to_graph(input2)
    assert "already exists in the graph" in str(exc_info.value)

def test_graph_builder_duplicate_node_name():
    """Test that ExecutableGraph prevents duplicate node names"""
    graph = (ExecutableGraph()
    .create_node(
        name="process",
        action_function=process_dataframe
    ))
    
    # Try to add second node with same name
    with pytest.raises(ValueError) as exc_info:
        graph.create_node(
            name="process",
            action_function=process_dataframe
        )
    assert "already exists in the graph" in str(exc_info.value) 