import pytest
from typing import Any
import pandas as pd
from pyautocausal.orchestration.nodes import Node, InputNode
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.persistence.output_config import OutputConfig
from pyautocausal.persistence.output_types import OutputType

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
    builder = GraphBuilder()
    
    # Create nodes
    builder.add_node(
        name="process_df",
        node=Node(
            name="process_df",
            action_function=process_dataframe,
            save_node=True
        )
    )
    builder.add_input_node(name="input_df")
    
    # Should not raise any errors
    node >> input_node

def test_incompatible_type_connection():
    """Test connecting nodes with incompatible types raises TypeError"""
    builder = GraphBuilder()
    
    # Create nodes with incompatible types
    node = Node(
        name="process_string",
        action_function=process_string,
        save_node=True
    )
    input_node = InputNode(name="input_df", input_type=pd.DataFrame)
    
    # Add to graph
    node.set_graph(builder.graph)
    input_node.set_graph(builder.graph)
    
    # Should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        node >> input_node
    assert "Type mismatch" in str(exc_info.value)

def test_untyped_function_warning(caplog):
    """Test warning is logged when connecting untyped function"""
    builder = GraphBuilder()
    
    # Create nodes
    node = Node(
        name="untyped",
        action_function=untyped_function,
        save_node=True
    )
    input_node = InputNode(name="input", input_type=Any)
    
    # Add to graph
    node.set_graph(builder.graph)
    input_node.set_graph(builder.graph)
    
    # Should log warning
    node >> input_node
    assert "Cannot validate connection" in caplog.text
    assert "lacks return type annotation" in caplog.text

def test_any_input_type_warning(caplog):
    """Test warning is logged when input node accepts Any type"""
    builder = GraphBuilder()
    
    # Create nodes
    node = Node(
        name="process_df",
        action_function=process_dataframe,
        save_node=True
    )
    input_node = InputNode(name="input", input_type=Any)
    
    # Add to graph
    node.set_graph(builder.graph)
    input_node.set_graph(builder.graph)
    
    # Should log warning
    node >> input_node
    assert "Cannot validate connection" in caplog.text
    assert "accepts Any type" in caplog.text

def test_chaining_connections():
    """Test chaining multiple connections"""
    builder = GraphBuilder()
    
    # Create nodes
    node1 = Node(
        name="process_df1",
        action_function=process_dataframe,
        save_node=True
    )
    node2 = Node(
        name="process_df2",
        action_function=process_dataframe,
        save_node=True
    )
    input_node = InputNode(name="input_df", input_type=pd.DataFrame)
    
    # Add to graph
    node1.set_graph(builder.graph)
    node2.set_graph(builder.graph)
    input_node.set_graph(builder.graph)
    
    # Chain connections
    node1 >> node2 >> input_node
    
    # Verify connections
    assert input_node in node2.get_successors()
    assert node2 in node1.get_successors() 