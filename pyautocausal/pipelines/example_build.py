from pyautocausal.orchestration.pipeline_graph import PipelineGraph
from pyautocausal.pipelines.library import PassthroughNode, OLSNode, DoubleMLNode
from dataclasses import dataclass
from pyautocausal.orchestration.data_input import DataInput
import pandas as pd
from pathlib import Path
from pyautocausal.persistence.local_output_handler import LocalOutputHandler

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
        
def test_data() -> pd.DataFrame:
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
    
if __name__ == "__main__":
    # Create pipeline
    Graph = CausalPipelineBuilder(output_path= Path("output"))
    Graph.add_branch(steps = [
        ('initial', PassthroughNode()),
        ('ols', OLSNode(condition = lambda: True)),
        ('end_first_branch', PassthroughNode())
    ])


    Graph.add_branch(steps = [
        ('doubleml', DoubleMLNode(condition = lambda: True)),
        ('end_second_branch', PassthroughNode())
    ], predecessor = 'initial')

    Graph.fit(test_data())


    

    
    