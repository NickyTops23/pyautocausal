from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import statsmodels.api as sm
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.pipelines.library import OLSNode, DoubleMLNode
from dataclasses import dataclass
from pyautocausal.orchestration.data_input import DataInput
from pyautocausal.orchestration.pipeline_graph import PipelineGraph


@dataclass
class ExampleCausalDataInput(DataInput):
    df: pd.DataFrame
    
    def to_dict(self) -> dict:
        return {
            "df": self.df
        }
    
    @classmethod
    def get_required_fields(cls) -> set[str]:
        return {"df"}
    
    @classmethod
    def check_presence_of_required_fields(cls) -> None:
        """Check if all required fields are present in input dictionary"""
        # No need to check to_dict() at class level
        # Just verify that the required fields are defined in the class
        required_fields = cls.get_required_fields()
        class_fields = {f.name for f in cls.__dataclass_fields__.values()}
        if not required_fields.issubset(class_fields):
            missing = required_fields - class_fields
            raise ValueError(f"Missing required fields: {missing}")

# -------------------------------------------------------------------------
# New CausalGraph class to hold our DoubleML and OLS nodes
# -------------------------------------------------------------------------
class ExampleCausalGraph(PipelineGraph):
    input_class = ExampleCausalDataInput
    
    def __init__(
            self, 
            input_data: ExampleCausalDataInput,
            output_path: Path
        ):
        super().__init__(
            input_data=input_data,
            output_handler=LocalOutputHandler(output_path),
        )

        # Define the DoubleML node
        self.doubleML_node = DoubleMLNode(
            graph=self,
            save_output=True,
            output_filename="doubleml_results"
        )
        self.doubleML_node.add_predecessor(self.starter_nodes["df"], argument_name="df")

        # Define the OLS node
        self.ols_node = OLSNode(
            graph=self,
            save_output=True,
            output_filename="ols_results"
        )
        self.ols_node.add_predecessor(self.starter_nodes["df"], argument_name="df")

    def add_data_node(self, load_data_node: Node):
        # Create a data distribution node
        self.data_distribution_node.add_predecessor(load_data_node, argument_name="df")

def load_lalonde_data() -> ExampleCausalDataInput:
    """
    Load and preprocess the LaLonde dataset.
    
    Returns:
        ExampleCausalDataInput: Preprocessed LaLonde dataset
    """
    url = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
    raw_df = pd.read_csv(url)
    
    # Preprocess into required format
    df = pd.DataFrame({
        'y': raw_df['re78'],
        'treat': raw_df['treat'],
        **raw_df.drop(columns=['re78', 'treat', 'ID']).to_dict('series')
    })
    
    return ExampleCausalDataInput(df=df)

# For convenience, if someone runs this module directly,
# we create 'output' folder and run the pipeline
if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    input_data = load_lalonde_data()
    
    # Create and execute graph
    graph = ExampleCausalGraph(
        input_data=input_data,
        output_path=path
    )
    graph.execute_graph()