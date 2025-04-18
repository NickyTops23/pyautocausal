import pytest
from pathlib import Path
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library import DoubleMLNode, OLSNode
from pyautocausal.pipelines.example import condition_nObs_DoubleML, condition_nObs_OLS

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

# Create reusable conditions
doubleml_condition = condition_nObs_DoubleML

ols_condition = condition_nObs_OLS

def causal_graph(output_dir):
    graph = (ExecutableGraph(output_path=output_dir)
        .create_input_node("df", input_dtype=pd.DataFrame )
        .create_decision_node(
            "doubleml_condition",
            doubleml_condition,
            predecessors=["df"],
        )
        .create_node(
            "doubleml",
            DoubleMLNode.action,
            predecessors=["doubleml_condition"],
            save_node=True,
        )
        .create_node(
            "ols",  
            OLSNode.action,
            predecessors=["doubleml_condition"],
            save_node=True,
        )
        .when_true("doubleml_condition", "doubleml")
        .when_false("doubleml_condition", "ols")
        )
    
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
    assert (output_dir / 'doubleml.txt').exists()
    assert not (output_dir / 'ols.txt').exists()

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
    assert (output_dir / 'ols.txt').exists()
    assert not (output_dir / 'doubleml.txt').exists()
