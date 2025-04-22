import pytest
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.estimators import fit_double_lasso, fit_ols
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.output import write_statsmodels_summary

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
