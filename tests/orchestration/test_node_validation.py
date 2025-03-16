import pytest
import pandas as pd
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.orchestration.nodes import Node, InputNode

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def test_duplicate_node_name():
    """Test that adding a node with a duplicate name raises an error"""
    graph = GraphBuilder().build()
    
    # Add first node
    node1 = Node(name="process", action_function=process_dataframe, graph=None)
    graph.add_node(node1)
    
    # Try to add second node with same name
    node2 = Node(name="process", action_function=process_dataframe, graph=None)
    with pytest.raises(ValueError) as exc_info:
        graph.add_node(node2)
    assert "already exists in the graph" in str(exc_info.value)

def test_duplicate_input_node_name():
    """Test that adding an input node with a duplicate name raises an error"""
    graph = GraphBuilder().build()
    
    # Add first input node
    input1 = InputNode(name="input", graph=None, input_dtype=pd.DataFrame)
    graph.add_input_node("input", input1)
    
    # Try to add second input node with same name
    input2 = InputNode(name="input", graph=None, input_dtype=pd.DataFrame)
    with pytest.raises(ValueError) as exc_info:
        graph.add_input_node("input", input2)
    assert "already exists" in str(exc_info.value)

def test_graph_builder_duplicate_node_name():
    """Test that GraphBuilder prevents duplicate node names"""
    builder = GraphBuilder()
    
    # Add first node
    builder.create_node(
        name="process",
        action_function=process_dataframe
    )
    
    # Try to add second node with same name
    with pytest.raises(ValueError) as exc_info:
        builder.create_node(
            name="process",
            action_function=process_dataframe
        )
    assert "already exists in the graph" in str(exc_info.value) 