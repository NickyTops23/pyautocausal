import pytest
from pathlib import Path
import pandas as pd
from pyautocausal.pipelines.example import ExampleCausalGraph, ExampleCausalDataInput

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

@pytest.fixture
def output_dir(tmp_path):
    """Create temporary directory for test outputs"""
    return tmp_path / "test_outputs"


def causal_graph(input_data, output_dir):
    input_data = ExampleCausalDataInput(df=input_data)
    """Create a CausalGraph instance with configured output handler"""
    output_dir.mkdir(parents=True, exist_ok=True)
    return ExampleCausalGraph(input_data=input_data, output_path=output_dir)

def test_causal_pipeline_execution(output_dir):
    """Test that the pipeline executes successfully"""
    
    causal_graph_lalonde = causal_graph(preprocess_lalonde_data(), output_dir)
    causal_graph_lalonde.execute_graph()
    
    # Check that output files are created
    expected_files = [
        'doubleml_results.txt',
        'ols_results.txt'
    ]
    
    # At least one of these files should exist based on data size
    assert any((output_dir / file).exists() for file in expected_files)

def test_causal_pipeline_large_dataset(output_dir):
    """Test pipeline with large dataset (should use DoubleML)"""
    # Create mock large dataset
    large_df = pd.DataFrame({
        'y': range(150),
        'treat': [1, 0] * 75,
        'age': range(150),
        'educ': [12] * 150,
    })
    
    causal_graph_large_df = causal_graph(large_df, output_dir)
    causal_graph_large_df.execute_graph()
    
    # Check that DoubleML results exist
    assert (output_dir / 'doubleml_results.txt').exists()
    assert not (output_dir / 'ols_results.txt').exists()

def test_causal_pipeline_small_dataset(output_dir):
    """Test pipeline with small dataset (should use OLS)"""
    # Create mock small dataset
    small_df = pd.DataFrame({
        'y': range(50),
        'treat': [1, 0] * 25,
        'age': range(50),
        'educ': [12] * 50,
    })
    
    causal_graph_small_df = causal_graph(small_df, output_dir)
    causal_graph_small_df.execute_graph()
    
    # Check that OLS results exist
    assert (output_dir / 'ols_results.txt').exists()
    assert not (output_dir / 'doubleml_results.txt').exists()


def test_causal_pipeline_invalid_path():
    """Test pipeline behavior with invalid output path"""
    invalid_path = Path('/nonexistent/directory')
    
    with pytest.raises(Exception):
        causal_graph(preprocess_lalonde_data(), invalid_path)



