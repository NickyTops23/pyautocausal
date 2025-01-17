from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import statsmodels.api as sm
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.pipelines.library import doubleML_treatment_effect, ols_treatment_effect, data_validation
from pyautocausal.persistence.output_config import OutputConfig, OutputType

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


def condition_nObs_DoubleML(df: pd.DataFrame) -> bool:
    return len(df) > 100

def condition_nObs_OLS(df: pd.DataFrame) -> bool:
    return len(df) <= 100

# -------------------------------------------------------------------------
# New CausalGraph class to hold our DoubleML and OLS nodes
# -------------------------------------------------------------------------
class CausalGraph(ExecutableGraph):
    def __init__(self, output_path: Path):
        super().__init__(output_handler=LocalOutputHandler(output_path))

        # Define the data distribution node
        self.data_distribution_node = Node(
            name="data_distribution",
            graph=self,
            action_function=data_validation,    
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
        self.doubleML_node.add_predecessor(self.data_distribution_node, argument_name="df")

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
        self.ols_node.add_predecessor(self.data_distribution_node, argument_name="df")

    def add_data_node(self, load_data_node: Node):
        # Create a data distribution node
        self.data_distribution_node.add_predecessor(load_data_node, argument_name="df")

# For convenience, if someone runs this module directly,
# we create 'output' folder and run the pipeline
if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    
    graph = CausalGraph(output_path= path)
    
    # Add the Lalonde data node (preprocessing)
    load_data_node = Node(
            name="load_data",
            graph=graph,
            action_function=preprocess_lalonde_data,
    )

    graph.add_data_node(load_data_node)
    # Execute all nodes
    graph.execute_graph()
