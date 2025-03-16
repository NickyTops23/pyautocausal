import pytest
import pandas as pd
import nbformat
from pathlib import Path
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.pipelines.library.models import OLSNode, PassthroughNode
import numpy as np

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
    graph = (GraphBuilder()
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
        .build())
    
    # Set input data
    input_data = pd.DataFrame({'value': [1, 2, 3]})
    graph.fit(data=input_data)
    
    return graph

@pytest.fixture
def causal_data():
    """Create a sample dataset suitable for causal inference"""
    return pd.DataFrame({
        'id_unit': range(100),
        't': [0] * 100,
        'treat': [1] * 50 + [0] * 50,
        'y': [i + j for i, j in zip(range(100), [2] * 50 + [0] * 50)],
        'x1': range(100),
        'x2': [i * 0.5 for i in range(100)]
    })

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

def test_static_method_export(causal_data, output_path):
    """Test exporting a graph with a static method."""
    # Create a graph with a PassthroughNode static method
    graph = (GraphBuilder()
             .add_input_node("df")
             .create_node(
                 name="passthrough_static",
                 action_function=PassthroughNode.action,
                 predecessors={'df': 'df'}
             )
             .build())
    
    # Fit the graph
    graph.fit(df=causal_data)
    
    # Export the notebook
    exporter = NotebookExporter(graph)
    exporter.export_notebook(str(output_path))
    
    # Check that the notebook file was created
    assert output_path.exists()
    
    # Load and verify notebook
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check that we have cells for the static method node
    node_cells = [
        cell for cell in nb.cells 
        if "passthrough_static" in cell.source
    ]
    
    # We should have at least 2 cells for the node (markdown + execution)
    assert len(node_cells) >= 2

def test_class_method_export(causal_data, output_path):
    """Test exporting a graph with a class instance method."""
    # Create a graph with an OLSNode instance method
    graph = (GraphBuilder()
             .add_input_node("df")
             .create_node(
                 name="ols_node",
                 action_function=OLSNode().action,
                 predecessors={'df': 'df'}
             )
             .build())
    
    # Fit the graph
    graph.fit(df=causal_data)
    
    # Export the notebook
    exporter = NotebookExporter(graph)
    exporter.export_notebook(str(output_path))
    
    # Check that the notebook file was created
    assert output_path.exists()
    
    # Load and verify notebook
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check that we have cells for the class method node
    node_cells = [
        cell for cell in nb.cells 
        if "ols_node" in cell.source
    ]
    
    # We should have at least 2 cells for the node (markdown + execution)
    assert len(node_cells) >= 2

def test_lambda_function_export(causal_data, output_path):
    """Test exporting a graph with a lambda function."""
    # Create a graph with a lambda function
    graph = (GraphBuilder()
             .add_input_node("df")
             .create_node(
                 name="lambda_node",
                 action_function=lambda df: df,
                 predecessors={'df': 'df'}
             )
             .build())
    
    # Fit the graph
    graph.fit(df=causal_data)
    
    # Export the notebook
    exporter = NotebookExporter(graph)
    exporter.export_notebook(str(output_path))
    
    # Check that the notebook file was created
    assert output_path.exists()
    
    # Load and verify notebook
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check that we have cells for the lambda function node
    node_cells = [
        cell for cell in nb.cells 
        if "lambda_node" in cell.source
    ]
    
    # We should have at least 2 cells for the node (markdown + execution)
    assert len(node_cells) >= 2
    
    # Check that the lambda was converted to a proper function
    lambda_def_cells = [
        cell for cell in nb.cells 
        if "lambda_node_func" in cell.source and "def" in cell.source
    ]
    
    assert len(lambda_def_cells) >= 1

def test_eval_function_raises_error(output_path):
    """Test that functions created with eval/exec raise errors during export."""
    # Create a function using exec - this will have no extractable source code
    function_code = "def eval_func(df): return df"
    namespace = {}
    exec(function_code, namespace)
    eval_func = namespace['eval_func']
    
    # Create a simple graph with this function
    graph = (GraphBuilder()
             .add_input_node("df")
             .create_node(
                 name="eval_node",
                 action_function=eval_func,
                 predecessors={'df': 'df'}
             )
             .build())
    
    # Fit the graph with a simple DataFrame
    graph.fit(df=pd.DataFrame({'a': [1, 2, 3]}))
    
    # The export should fail because the function's source can't be extracted
    exporter = NotebookExporter(graph)
    
    # This should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        exporter.export_notebook(str(output_path))
    
    # Check that the error message mentions source code extraction
    assert "Could not extract source code" in str(excinfo.value)
    assert "eval_node" in str(excinfo.value)