from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import statsmodels.api as sm
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.pipelines.library import doubleML_treatment_effect, ols_treatment_effect, data_validation
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from dataclasses import dataclass
from ..orchestration.data_input import DataInput
from ..orchestration.pipeline_graph import PipelineGraph


def condition_nObs_DoubleML(df: pd.DataFrame) -> bool:
    return len(df) > 100

def condition_nObs_OLS(df: pd.DataFrame) -> bool:
    return len(df) <= 100

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
        self.doubleML_node = Node(
            name="doubleML_treatment_effect",
            graph=self,
            action_function=doubleML_treatment_effect,
            condition=condition_nObs_DoubleML,
            skip_reason="Sample size too small for Double ML",
            output_config=OutputConfig(
                save_output=True,
                output_filename="doubleml_results",
                output_type=OutputType.TEXT
            )
        )
        self.doubleML_node.add_predecessor(self.starter_nodes["df"], argument_name="df")

        # Define the OLS node
        self.ols_node = Node(
            name="ols_treatment_effect",
            graph=self,
            action_function=ols_treatment_effect,
            condition=condition_nObs_OLS,
            skip_reason="Sample size too large for OLS",
            output_config=OutputConfig(
                save_output=True,
                output_filename="ols_results",
                output_type=OutputType.TEXT
            )
        )
        self.ols_node.add_predecessor(self.starter_nodes["df"], argument_name="df")

    def add_data_node(self, load_data_node: Node):
        # Create a data distribution node
        self.data_distribution_node.add_predecessor(load_data_node, argument_name="df")

# For convenience, if someone runs this module directly,
# we create 'output' folder and run the pipeline
if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    
    graph = ExampleCausalGraph(output_path= path)
    
    # Add the Lalonde data node (preprocessing)
    load_data_node = Node(
            name="load_data",
            graph=graph,
            action_function=preprocess_lalonde_data,
    )

    graph.add_data_node(load_data_node)
    # Execute all nodes
    graph.execute_graph()
