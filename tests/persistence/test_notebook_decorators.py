import pytest
import pandas as pd
import numpy as np
from pyautocausal.persistence.notebook_decorators import expose_in_notebook
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType
import os
import nbformat

# Define a complex function that we want to expose in the notebook
def complex_statistical_function(df: pd.DataFrame, alpha: float = 0.05):
    """
    Perform a complex statistical analysis on the dataframe.
    
    Args:
        df: The input DataFrame to analyze
        alpha: Significance level
        
    Returns:
        Summary statistics
    """
    # Calculate mean and standard deviation for each column
    means = df.mean()
    stds = df.std()
    
    # Calculate confidence intervals
    z_value = 1.96  # Approx. 95% confidence
    ci_lower = means - z_value * stds / np.sqrt(len(df))
    ci_upper = means + z_value * stds / np.sqrt(len(df))
    
    return pd.DataFrame({
        'mean': means,
        'std': stds,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

# Create a wrapper function that adapts parameter names
@expose_in_notebook(
    target_function=complex_statistical_function,
    arg_mapping={'data': 'df', 'significance': 'alpha'}
)
def stats_wrapper(data, significance=0.05):
    """Wrapper for complex_statistical_function that adapts parameter names."""
    return complex_statistical_function(df=data, alpha=significance)

# Test that the wrapper function works and maintains its behavior
def test_wrapper_execution():
    """Test that the wrapper function executes correctly."""
    # Create test data
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    # Call the wrapper function
    result = stats_wrapper(test_data, significance=0.01)
    
    # Check that we got expected output
    assert isinstance(result, pd.DataFrame)
    assert 'mean' in result.columns
    assert 'std' in result.columns
    assert 'ci_lower' in result.columns
    assert 'ci_upper' in result.columns

# Test the notebook export functionality with the decorated function
@pytest.fixture
def sample_graph(tmp_path):
    """Create a sample graph with a wrapper function."""
    # Create a test DataFrame
    def create_data() -> pd.DataFrame:
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    # Build a simple graph
    graph = (ExecutableGraph(output_path=tmp_path)
        .create_node(
            "data",
            create_data,
        )
        .create_node(
            "stats_processor",
            stats_wrapper,
            predecessors=["data"],
        )
    )
    
    # Execute the graph
    graph.execute_graph()
    return graph

def test_notebook_export_with_wrapper(sample_graph, tmp_path):
    """Test that the notebook exporter correctly handles decorated wrapper functions."""
    # Export the notebook
    notebook_path = os.path.join(tmp_path, "test_notebook.ipynb")
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(notebook_path)
    
    # Verify the notebook exists
    assert os.path.exists(notebook_path)
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Look for key indicators that our wrapper was properly processed
    found_target_function = False
    found_wrapper_function = False
    found_arg_mapping = False
    found_direct_call = False
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if 'complex_statistical_function' in cell.source:
                found_target_function = True
            if 'stats_wrapper' in cell.source:
                found_wrapper_function = True
            if "Argument mapping: 'data' → 'df'" in cell.source:
                found_arg_mapping = True
            if "Alternatively, call the target function directly" in cell.source:
                found_direct_call = True
    
    # Verify we found all the expected elements
    assert found_target_function, "Target function not found in notebook"
    assert found_wrapper_function, "Wrapper function not found in notebook"
    assert found_arg_mapping, "Argument mapping not found in notebook"
    assert found_direct_call, "Direct call alternative not found in notebook" 