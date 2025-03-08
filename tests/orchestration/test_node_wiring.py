import pytest
from typing import Any
import pandas as pd
from pyautocausal.orchestration.nodes import Node, InputNode
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.persistence.output_config import OutputConfig
from pyautocausal.persistence.output_types import OutputType
from unittest.mock import Mock


# Test functions with type hints
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def process_string(s: str) -> str:
    return s

def untyped_function(x):
    return x

# Test cases
def test_valid_type_connection():
    """Test connecting nodes with compatible types"""
    graph = GraphBuilder().build()
    
    # Create nodes
    node = Node(
        name="process_df",
        action_function=process_dataframe,
        save_node=True,
        graph=graph
    )
    input_node = InputNode(name="input_df", graph=graph, input_dtype=pd.DataFrame)
    
    # Should not raise any errors
    node >> input_node

    assert input_node in node.get_successors()
    assert node in input_node.get_predecessors()

def test_incompatible_type_connection():
    """Test connecting nodes with incompatible types raises TypeError"""
    graph = Mock()
    
    # Create nodes with incompatible types
    node = Node(
        name="process_string",
        action_function=process_string,
        save_node=True,
        graph=graph
    )
    input_node = InputNode(name="input_df", graph=graph, input_dtype=pd.DataFrame)
    
    # Should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        node >> input_node
    assert "Type mismatch" in str(exc_info.value)

def test_untyped_function_warning():
    """Test warning is logged when connecting untyped function"""
    graph = Mock()
    
    # Create nodes
    node = Node(
        name="untyped",
        action_function=untyped_function,
        save_node=False,
        graph=graph
    )
    input_node = InputNode(name="input", graph=graph, input_dtype=Any)
    
    # Should log warning
    with pytest.warns(Warning) as warning_info:
        node >> input_node
        assert "Cannot validate connection" in str(warning_info[0].message)
        assert "lacks return type annotation" in str(warning_info[0].message)

def test_any_input_type_warning():
    """Test warning is logged when input node accepts Any type"""
    graph = Mock()
    
    # Create nodes
    node = Node(
        name="process_df",
        action_function=process_dataframe,
        save_node=True,
        graph=graph
    )
    input_node = InputNode(name="input", graph=graph, input_dtype=Any)
    
    # Should log warning
    with pytest.warns(Warning) as warning_info:
        node >> input_node
        assert "Cannot validate connection" in str(warning_info[0].message)
        assert "accepts Any type" in str(warning_info[0].message)

def test_chaining_connections():
    """Test chaining multiple connections"""
    graph = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create nodes
    node1 = Node(
        name="process_df1",
        action_function=process_dataframe,
        save_node=False,
        graph=graph
    )

    

    input_node = InputNode(name="input_df", graph=graph2, input_dtype=pd.DataFrame)
    input_node2 = InputNode(name="input_df2", graph=graph2, input_dtype=pd.DataFrame)
    
    # Chain connections
    node1 >> input_node
    node1 >> input_node2

    
    # Verify connections
    assert input_node in node1.get_successors()
    assert input_node2 in node1.get_successors() 
