import pytest
import pandas as pd
from pathlib import Path
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.pipelines.library import OLSNode, DoubleMLNode, PassthroughNode
from pyautocausal.persistence.local_output_handler import LocalOutputHandler

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'y': range(20),
        'treat': [0, 1] * 10,
        'x1': range(20),
        'x2': range(20, 40)
    })

@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "test_outputs"

def test_create_simple_pipeline(sample_df, output_path):
    # Create builder with output handler
    builder = GraphBuilder(output_path=output_path)
    
    # Create input node
    builder.add_input_node("data")
    
    # Create processing nodes
    ols_node = OLSNode()
    # Build graph
    builder.add_node("ols", ols_node, predecessors={"df": "data"})
    
    # Build and execute graph
    graph = builder.build()
    graph.fit(data=sample_df)
    
    # Get nodes from graph to check outputs
    ols_node = [n for n in graph.nodes() if isinstance(n, OLSNode)][0]    
    # Verify outputs exist
    assert ols_node.output is not None
    assert (output_path / "ols_treatment_effect.txt").exists()


def test_custom_conditions(sample_df, output_path):
    # Create builder with output handler
    builder = GraphBuilder(output_path=output_path)
    
    # Create input node
    builder.add_input_node("data")
    
    # Create processing nodes
    ols_node = OLSNode(
        condition=lambda df: len(df) > 25,  # Should skip since sample_df has 20 rows
        skip_reason="Sample size is too small"
    )
    
    # Build graph
    builder.add_node("ols", ols_node, predecessors={"df": "data"})
    
    # Build and execute graph
    graph = builder.build()
    graph.fit(data=sample_df)
    
    # Get nodes from graph to check outputs
    ols_node = [n for n in graph.nodes() if isinstance(n, OLSNode)][0]    
    
    # Verify node was skipped
    assert ols_node.is_skipped()
    # Verify output file was not created since node was skipped
    assert not (output_path / "ols_treatment_effect.txt").exists()
