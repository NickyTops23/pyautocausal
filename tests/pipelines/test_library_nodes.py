import pytest
import pandas as pd
from pathlib import Path
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library import OLSNode, DoubleMLNode, PassthroughNode
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.orchestration.condition import Condition
from pyautocausal.persistence.output_config import OutputConfig, OutputType

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with 20 rows"""
    return pd.DataFrame({
        'y': range(20),
        'treat': [1, 0] * 10,
        'x': range(20)
    })

@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "test_outputs"

def test_create_simple_pipeline(sample_df, output_path):
    """Test basic node execution with default settings"""
    
    def ols_action(data: pd.DataFrame) -> pd.Series:
        return OLSNode.action(data)
    
    # Build graph using builder pattern
    graph = (ExecutableGraph(output_path=output_path)
        .add_input_node("data")
        .create_node(
            "ols",
            ols_action,
            predecessors=["data"],
            save_node=True,
            output_config=OutputConfig(
                output_filename="ols_treatment_effect",
                output_type=OutputType.TEXT
            )
        )
        )
    
    # Execute graph
    graph.fit(data=sample_df)
    
    # Get node from graph
    ols_node = [n for n in graph.nodes() if n.name == "ols"][0]
    
    # Verify outputs exist
    assert ols_node.output is not None
    assert (output_path / "ols_treatment_effect.txt").exists()

def test_custom_conditions(sample_df, output_path):
    """Test that nodes respect custom conditions"""
    # Create condition that skips when df has less than 25 rows
    small_data_condition = Condition(
        lambda df: len(df) > 25,
        "Sample size is too small"
    )
    
    # Build graph using builder pattern
    graph = (ExecutableGraph(output_path=output_path)
        .add_input_node("data")
        .create_node(
            "ols",
            OLSNode.action,
            predecessors=["data"],
            condition=small_data_condition,
            save_node=True,
            output_config=OutputConfig(
                output_filename="ols_treatment_effect",
                output_type=OutputType.TEXT
            )
        )
        )
    
    # Execute graph
    graph.fit(data=sample_df)
    
    # Get node from graph
    ols_node = [n for n in graph.nodes() if n.name == "ols"][0]
    
    # Verify node was skipped due to condition
    assert ols_node.is_skipped()
    
    # Verify output file was not created since node was skipped
    assert not (output_path / "ols_treatment_effect.txt").exists()
