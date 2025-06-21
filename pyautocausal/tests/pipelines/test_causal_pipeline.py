import pytest
import pandas as pd
from pathlib import Path
import os
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.estimators import fit_double_lasso, fit_ols
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.output import write_statsmodels_summary
from pyautocausal.pipelines.example_graph import simple_graph, generate_mock_data
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.persistence.visualizer import visualize_graph
import json
import nbformat
import logging
from pyautocausal.orchestration.nodes import NodeState

def preprocess_lalonde_data() -> str:
    """
    Load and preprocess the LaLonde dataset.
    
    Returns:
        str: String representation of the processed dataset
    """
    url = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
    df = pd.read_csv(url)
    y = df['re78']
    t = df['treat']
    X = df.drop(columns=['re78', 'treat','ID'])

    df = pd.DataFrame({'y': y, 'treat': t, **X})
    return df

# The causal_graph function is now imported from conftest.py

def test_causal_pipeline_execution(causal_graph):
    """Test that the pipeline executes successfully"""
    
    # Execute with input data
    causal_graph.fit(data=preprocess_lalonde_data())

def test_causal_pipeline_large_dataset(causal_graph, tmp_path):
    """Test pipeline with large dataset (should use DoubleML)"""
    # Create mock large dataset
    large_df = pd.DataFrame({
        'y': range(150),
        'treat': [1, 0] * 75,
        'age': range(150),
        'educ': [12] * 150,
    })
    
    causal_graph.fit(data=large_df)
    
    # Get the output directory from the graph
    output_dir = tmp_path / "causal_output"
    
    # Check that DoubleML results exist
    assert (output_dir / 'doubleml_summary.txt').exists()
    assert not (output_dir / 'ols_summary.txt').exists()

def test_causal_pipeline_small_dataset(causal_graph, tmp_path):
    """Test pipeline with small dataset (should use OLS)"""
    # Create mock small dataset
    small_df = pd.DataFrame({
        'y': range(50),
        'treat': [1, 0] * 25,
        'age': range(50),
        'educ': [12] * 50,
    })
    
    causal_graph.fit(data=small_df)
    
    # Get the output directory from the graph
    output_dir = tmp_path / "causal_output"
    
    # Check that OLS results exist
    assert (output_dir / 'ols_summary.txt').exists()
    assert not (output_dir / 'doubleml_summary.txt').exists()

def test_example_graph_pipeline(tmp_path):
    """Test that the example graph pipeline executes successfully and produces expected outputs."""
    # Create output directory
    output_path = tmp_path / "example_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the graph
    graph = simple_graph(output_path)
    
    # Generate mock data with more units for proper synthetic control
    data = generate_mock_data(n_units=2000, n_periods=2, n_treated=500)
    data_path = output_path / "simple_graph_data.csv"
    data.to_csv(data_path, index=False)
    
    # Fit the graph
    graph.fit(df=data)
    
    # Check node states
    completed_nodes = 0
    for node in graph.nodes():
        if hasattr(node, 'state'):
            assert node.state is not None, f"Node {node.name} has no state"
            if hasattr(node.state, 'name') and node.state.name == 'COMPLETED':
                completed_nodes += 1
    
    # Ensure at least some nodes completed successfully
    assert completed_nodes > 0, "No nodes completed successfully"
    
    # Test visualization
    viz_path = output_path / "simple_graph.md"
    visualize_graph(graph, save_path=str(viz_path))
    assert viz_path.exists(), "Graph visualization was not created"
    
    # Test notebook export
    notebook_path = output_path / "simple_graph.ipynb"
    exporter = NotebookExporter(graph)
    exporter.export_notebook(notebook_path)
    assert notebook_path.exists(), "Notebook was not exported"

    # Load and parse the notebook properly
    with open(notebook_path, 'r') as f:
        notebook_json = json.load(f)
        notebook = nbformat.reads(json.dumps(notebook_json), as_version=4)
    
    import_cell_contents = "\n".join(exporter.needed_imports)
    # Now properly check cells
    found_import_cell = False
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell_source = cell.source
            if cell_source == import_cell_contents:
                found_import_cell = True
                break

    assert found_import_cell, "Notebook does not contain the expected import cell"

    # Verify notebook content
    with open(notebook_path, 'r') as f:
        notebook_content = f.read()
        assert "Causal Analysis Pipeline" in notebook_content, "Notebook header missing"
        
    # Check for any expected output files from the graph execution
    output_files = list(output_path.glob("*.csv")) + list(output_path.glob("*.txt"))
    assert len(output_files) > 1, "Expected output files were not generated"

def test_serialized_simple_graph_execution(tmp_path):
    """
    Tests that a serialized and deserialized graph can be executed successfully.
    This provides an end-to-end check of the YAML round-trip and execution logic.
    """
    # 1. Build the initial graph and configure its output path
    graph_output_path = tmp_path / "original_output"
    graph = simple_graph(graph_output_path)
    
    # 2. Serialize the graph to a YAML file
    yaml_path = tmp_path / "simple_graph.yml"
    graph.to_yaml(yaml_path)
    assert yaml_path.exists()

    # 3. Load the graph back from the YAML file
    loaded_graph = ExecutableGraph.from_yaml(yaml_path)
    
    # Configure the runtime for the deserialized graph
    loaded_output_path = tmp_path / "loaded_output"
    loaded_graph.configure_runtime(output_path=loaded_output_path)

    # 4. Execute the loaded graph with mock data
    mock_data = generate_mock_data(n_units=50, n_periods=2, n_treated=10)
    loaded_graph.fit(df=mock_data)

    # 5. Assert that the execution was successful
    # Check that all nodes reached a terminal state
    for node in loaded_graph.nodes():
        if hasattr(node, 'state'):
            assert node.state.is_terminal(), f"Node {node.name} did not complete successfully. State: {node.state.name}"

    # 5. Verify that output files were created for the executed branch.
    #    The mock data has multiple periods, so the 'did_spec' branch should run.
    did_node = loaded_graph.get("save_ols_did")
    stand_node = loaded_graph.get("ols_stand_output")

    # The stand_spec branch should be PASSED (skipped)
    assert stand_node.state == NodeState.PASSED, "Standard spec branch should have been skipped."
    assert not (loaded_output_path / "ols_stand_output.txt").exists(), "Output file from skipped branch should not exist."
    
    # The DiD branch should be COMPLETED
    assert did_node.state == NodeState.COMPLETED, "DiD spec branch should have completed."
    assert (loaded_output_path / "save_ols_did.txt").exists(), "Output file from executed DiD branch should exist."

    # Export the notebook
    notebook_path = loaded_output_path / "simple_graph.ipynb"
    exporter = NotebookExporter(loaded_graph)
    exporter.export_notebook(notebook_path)
    assert notebook_path.exists(), "Notebook was not exported"
    
    # Load and parse the notebook properly
    with open(notebook_path, 'r') as f:
        notebook_json = json.load(f)
        notebook = nbformat.reads(json.dumps(notebook_json), as_version=4)
    
    import_cell_contents = "\n".join(exporter.needed_imports)
    # Now properly check cells
    found_import_cell = False
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell_source = cell.source
            if cell_source == import_cell_contents:
                found_import_cell = True
                break

    assert found_import_cell, "Notebook does not contain the expected import cell"

    # Verify notebook content
    with open(notebook_path, 'r') as f:
        notebook_content = f.read()
        assert "Causal Analysis Pipeline" in notebook_content, "Notebook header missing"