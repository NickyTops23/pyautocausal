import pytest
import pandas as pd
import nbformat
from pathlib import Path
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.orchestration.graph import ExecutableGraph

def simple_func(x: pd.DataFrame) -> pd.DataFrame:
    """A simple function that adds a column"""
    x['new_col'] = x['value'] * 2
    return x

def process_func(df: pd.DataFrame) -> str:
    """Process the data and return a string"""
    return f"Processed {len(df)} rows"

@pytest.fixture
def sample_graph():
    """Create a simple graph for testing"""
    graph = (ExecutableGraph()
        .add_input_node("data")
        .create_node(
            "transform",
            simple_func,
            predecessors={"x": "data"}
        )
        .create_node(
            "process",
            process_func,
            predecessors={"df": "transform"}
        )
        )
    
    # Set input data
    input_data = pd.DataFrame({'value': [1, 2, 3]})
    graph.fit(data=input_data)
    
    return graph

@pytest.fixture
def output_path(tmp_path):
    """Create temporary directory for notebook output"""
    return tmp_path / "test_notebook.ipynb"

def test_notebook_creation(sample_graph, output_path):
    """Test basic notebook creation"""
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(output_path))
    
    assert output_path.exists()
    
    # Load and verify notebook
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check basic structure
    assert len(nb.cells) > 0
    assert nb.cells[0].cell_type == "markdown"  # Header
    assert nb.cells[1].cell_type == "code"      # Imports

def test_node_cell_creation(sample_graph, output_path):
    """Test that cells are created for each node"""
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(output_path))
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Count node-related cells (each node should have at least 2 cells)
    node_cells = [
        cell for cell in nb.cells 
        if any(node.name in cell.source for node in sample_graph.nodes())
    ]
    
    # We should have at least 2 cells per node (markdown + execution)
    assert len(node_cells) >= len(sample_graph.nodes()) * 2

def test_topological_order(sample_graph, output_path):
    """Test that nodes appear in topological order"""
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(output_path))
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Extract node names in order of appearance
    node_order = []
    for cell in nb.cells:
        for node in sample_graph.nodes():
            if node.name in cell.source and node.name not in node_order:
                node_order.append(node.name)
    
    # Verify order (data should come before transform, which comes before process)
    assert node_order.index('data') < node_order.index('transform')
    assert node_order.index('transform') < node_order.index('process')