from pyautocausal.orchestration.pipeline_graph import PipelineGraph
from pyautocausal.pipelines.library import PassthroughNode, OLSNode, DoubleMLNode
from dataclasses import dataclass
from pyautocausal.orchestration.data_input import DataInput
import pandas as pd
from pathlib import Path
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
import pytest

@dataclass
class CausalDataInput(DataInput):
    df: pd.DataFrame
    
    def to_dict(self) -> dict:
        return {
            "df": self.df
        }
    
    @classmethod
    def get_required_fields(cls) -> set[str]:
        return {"df"}
    
class CausalPipelineBuilder(PipelineGraph):
    input_class = CausalDataInput

    def __init__(
            self, 
            output_path: Path
        ):
        super().__init__(
            output_handler=LocalOutputHandler(output_path),
        )
        
@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Create a sample dataset mimicking LaLonde format for testing.
    
    Returns:
        pd.DataFrame: Sample dataset with treatment effect structure
    """
    # Create sample data
    return pd.DataFrame({
        'y': range(100),  # outcome
        'treat': [1, 0] * 50,  # treatment indicator
        'age': range(100),  # covariate
        'educ': [12] * 100  # covariate
    })

@pytest.fixture(autouse=True)
def cleanup(tmp_path):
    """Cleanup output directory after each test"""
    yield
    output_dir = tmp_path / "output"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

def test_pipelinebuilder(tmp_path):
    """
    Test whether the pipeline added all nodes
    """
    Graph = CausalPipelineBuilder(output_path=tmp_path / "output")

    # Define pipeline structure - no need to set graph when creating nodes
    Graph.add_branch(steps = [
        ('initial', PassthroughNode()),
        ('ols', OLSNode(condition = lambda: True)),
        ('end', PassthroughNode())
    ])
    
    assert 'initial' in Graph.node_index
    assert 'ols' in Graph.node_index
    assert 'end' in Graph.node_index
    assert Graph.node_index['initial'].get_predecessors() == set()
    assert [i.name for i in Graph.node_index['ols'].get_predecessors()] == ['initial']
    assert [i.name for i in Graph.node_index['end'].get_predecessors()] == ['ols']


def test_duplicate_node_error(tmp_path):
    """Test whether an error is raised when we add duplicate nodes"""
    Graph = CausalPipelineBuilder(output_path=tmp_path / "output")
    Graph.add_branch(steps = [
        ('initial', PassthroughNode()),
        ('ols', OLSNode(condition = lambda: True)),
        ('end', PassthroughNode())
    ])
    
    with pytest.raises(ValueError):
        Graph.add_branch(steps = [
            ('initial', PassthroughNode())
        ])

def test_adding_predecessor_to_nonexistent_node(tmp_path):
    """Test adding a predecessor to a nonexistent node"""
    Graph = CausalPipelineBuilder(output_path=tmp_path / "output")

    with pytest.raises(ValueError):
         Graph.add_branch(steps = [
            ('initial', PassthroughNode())
        ], predecessor = 'nonexistent')
        
def test_adding_predecessor_to_existing_node(tmp_path):
    """Test adding a predecessor to an existing node"""
    Graph = CausalPipelineBuilder(output_path=tmp_path / "outputs")
    Graph.add_branch(steps = [
        ('initial', PassthroughNode())
    ])
    Graph.add_branch(steps = [
        ('ols', OLSNode(condition = lambda: True))
    ], predecessor = 'initial')

    assert [i.name for i in Graph.node_index['ols'].get_predecessors()] == ['initial']

def test_fitting_pipeline(tmp_path, sample_data):
    """Test fitting the pipeline"""
    output_dir = tmp_path / "output"

    Graph = CausalPipelineBuilder(output_path=output_dir)
    Graph.add_branch(steps = [
        ('initial', PassthroughNode()),
        ('ols', OLSNode(condition = lambda: True)),
        ('end', PassthroughNode())
    ])
    Graph.fit(sample_data)
    
    assert output_dir.exists()
    assert (output_dir / 'ols_treatment_effect.txt').exists()
    assert (output_dir / 'ols_treatment_effect.txt').stat().st_size > 0
    assert Graph.node_index['initial'].output is not None
    assert Graph.node_index['ols'].output is not None
    assert Graph.node_index['end'].output is not None
