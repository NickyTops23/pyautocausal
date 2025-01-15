from pathlib import Path
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
import pandas as pd
import statsmodels.api as sm
from pyautocausal.pipelines.library import doubleML_treatment_effect, ols_treatment_effect
from pyautocausal.persistence.output_config import OutputConfig, OutputType

def preprocess_lalonde_data() -> pd.DataFrame:
    """
    Load and preprocess the LaLonde dataset.
    
    Returns:
        pd.DataFrame: Processed dataset with columns:
            - y: Outcome variable (re78)
            - t: Treatment indicator
            - Additional covariates from original dataset
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

def causal_pipeline(path: Path) -> None:
    """
    Execute a conditional pipeline for treatment effect estimation.
    
    Uses Double ML for large samples (>100 observations) and 
    OLS for smaller samples.

    Args:
        path (Path): Directory where outputs will be saved
    """
    graph = ExecutableGraph(output_handler=LocalOutputHandler(path))
    
    load_data_node = Node(
        "load_data", 
        graph, 
        preprocess_lalonde_data
    )

    doubleML_node = Node(
        name="doubleML_treatment_effect", 
        graph=graph, 
        action_function=doubleML_treatment_effect,
        condition=condition_nObs_DoubleML,
        skip_reason="Sample size too small for Double ML",
        output_config=OutputConfig(
            save_output=True,
            output_filename="doubleml_results",
            output_type=OutputType.TEXT
        )
    )

    ols_node = Node(
        name="ols_treatment_effect", 
        graph=graph, 
        action_function=ols_treatment_effect,
        condition=condition_nObs_OLS,
        skip_reason="Sample size too large for OLS",
        output_config=OutputConfig(
            save_output=True,
            output_filename="ols_results",
            output_type=OutputType.TEXT
        )
    )
    
    doubleML_node.add_predecessor(load_data_node, argument_name="df")
    ols_node.add_predecessor(load_data_node, argument_name="df")
    
    graph.execute_graph()

if __name__ == "__main__":
    path = Path("output")
    path.mkdir(parents=True, exist_ok=True)
    causal_pipeline(path)
