import pytest
from pathlib import Path
import pandas as pd
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.pipelines.library import OLSNode, DoubleMLNode

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

def causal_graph(output_dir):
    graph = (GraphBuilder(output_path=output_dir)
        .add_input_node("df")
        .create_node(
            "doubleml",
            DoubleMLNode.action,
            predecessors={"df": "df"},
            condition=DoubleMLNode.condition,
            skip_reason="Sample size too small for Double ML",
            output_config=OutputConfig(
                save_output=True,
                output_filename="doubleml_results",
                output_type=OutputType.TEXT
            )
        )
        .create_node(
            "ols",
            OLSNode.action,
            predecessors={"df": "df"},
            condition=OLSNode.condition,
            skip_reason="Sample size too large for OLS",
            output_config=OutputConfig(
                save_output=True,
                output_filename="ols_results",
                output_type=OutputType.TEXT
            )
        )
        .build())
    
    return graph

def test_causal_pipeline_execution(output_dir):
    """Test that the pipeline executes successfully"""
    
    graph = causal_graph(output_dir)
    
    # Execute with input data
    graph.fit(df=preprocess_lalonde_data())

def test_causal_pipeline_large_dataset(output_dir):
    """Test pipeline with large dataset (should use DoubleML)"""
    # Create mock large dataset
    large_df = pd.DataFrame({
        'y': range(150),
        'treat': [1, 0] * 75,
        'age': range(150),
        'educ': [12] * 150,
    })
    
    graph = causal_graph(output_dir)
    graph.fit(df=large_df)
    
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
    
    graph = causal_graph(output_dir)
    graph.fit(df=small_df)
    
    # Check that OLS results exist
    assert (output_dir / 'ols_results.txt').exists()
    assert not (output_dir / 'doubleml_results.txt').exists()
