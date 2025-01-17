import pytest
from pathlib import Path
import pandas as pd
from pyautocausal.pipelines.example import CausalGraph, preprocess_lalonde_data
from pyautocausal.orchestration.nodes import Node

@pytest.fixture
def output_dir(tmp_path):
    """Create temporary directory for test outputs"""
    return tmp_path / "test_outputs"

@pytest.fixture
def causal_graph(output_dir):
    """Create a CausalGraph instance with configured output handler"""
    output_dir.mkdir(parents=True, exist_ok=True)
    return CausalGraph(output_path=output_dir)

def test_causal_pipeline_execution(causal_graph, output_dir):
    """Test that the pipeline executes successfully"""
    # Add the data loading node
    load_data_node = Node(
        name="load_data",
        graph=causal_graph,
        action_function=preprocess_lalonde_data,
    )
    
    causal_graph.add_data_node(load_data_node)
    causal_graph.execute_graph()
    
    # Check that output files are created
    expected_files = [
        'doubleml_results.txt',
        'ols_results.txt'
    ]
    
    # At least one of these files should exist based on data size
    assert any((output_dir / file).exists() for file in expected_files)

def test_causal_pipeline_small_dataset(causal_graph, output_dir):
    """Test pipeline with large dataset (should use DoubleML)"""
    # Create mock large dataset
    large_df = pd.DataFrame({
        'y': range(150),
        'treat': [1, 0] * 20,
        'age': range(40),
        'educ': [12] * 40,
    })
    
    # Mock the preprocess function to return large dataset
    mock_data_node = Node(
        name="load_data",
        graph=causal_graph,
        action_function=lambda: large_df,
    )
    
    causal_graph.add_data_node(mock_data_node)
    causal_graph.execute_graph()
    
    # Check that DoubleML results exist
    assert (output_dir / 'doubleml_results.txt').exists()
    assert not (output_dir / 'ols_results.txt').exists()

def test_causal_pipeline_small_dataset(causal_graph, output_dir):
    """Test pipeline with small dataset (should use OLS)"""
    # Create mock small dataset
    small_df = pd.DataFrame({
        'y': range(50),
        'treat': [1, 0] * 25,
        'age': range(50),
        'educ': [12] * 50,
    })
    
    # Mock the preprocess function to return small dataset
    mock_data_node = Node(
        name="load_data",
        graph=causal_graph,
        action_function=lambda: small_df,
    )
    
    causal_graph.add_data_node(mock_data_node)
    causal_graph.execute_graph()
    
    # Check that OLS results exist
    assert (output_dir / 'ols_results.txt').exists()
    assert not (output_dir / 'doubleml_results.txt').exists()


def test_causal_pipeline_invalid_path():
    """Test pipeline behavior with invalid output path"""
    invalid_path = Path('/nonexistent/directory')
    
    with pytest.raises(Exception):
        CausalGraph(output_path=invalid_path)



