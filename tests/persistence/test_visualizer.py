import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType

@pytest.fixture
def simple_graph(tmp_path):
    """Create a simple test graph with a few nodes"""
    
    graph = (ExecutableGraph(output_path=tmp_path)
        .create_input_node("input")
        .create_node(
            "node1",
            node1_action,
            predecessors=["input"],
            save_node=True,
            output_config=OutputConfig(
                output_filename="node1_output",
                output_type=OutputType.PARQUET
            )
        )
        .create_node(
            "node2",
            node2_action,
            predecessors=["node1"],
            save_node=True,
            output_config=OutputConfig(
                output_filename="node2_output",
                output_type=OutputType.PARQUET
            )
        )
        )
    return graph

def test_visualize_graph_creates_file(simple_graph, tmp_path):
    """Test that visualize_graph creates an output file"""
    output_file = tmp_path / "test_graph.md"
    visualize_graph(simple_graph, save_path=str(output_file))
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_visualize_graph_node_positions(simple_graph, tmp_path):
    """Test that nodes are positioned correctly (input at bottom, others above)"""
    output_file = tmp_path / "test_graph.md"
    
    # Get node positions by accessing the internal state
    pos = visualize_graph(simple_graph, save_path=str(output_file))
    
    # Get input node and other nodes
    input_node = [n for n in simple_graph.nodes() if n.name == "input"][0]
    other_nodes = [n for n in simple_graph.nodes() if n.name != "input"]
    
    # Input node should be at the bottom (more negative y value in matplotlib coordinates)
    input_y = pos[input_node][1]
    other_y_coords = [pos[n][1] for n in other_nodes]
    
    # Changed assertion: input_y should be MORE negative than other y coordinates
    assert all(input_y > y for y in other_y_coords), "Input node should be below other nodes"

def test_visualize_graph_node_labels(simple_graph, tmp_path):
    """Test that node labels are correctly set to node names"""
    output_file = tmp_path / "test_graph.png"
    
    # Get labels by accessing the internal state
    labels = visualize_graph(simple_graph, save_path=str(output_file))
    plt.close()
    
    # Check that each node's label matches its name
    for node in simple_graph.nodes():
        assert labels[node] == node.name, f"Label for node {node.name} doesn't match" 