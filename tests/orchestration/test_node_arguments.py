import pytest
import pandas as pd
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.run_context import RunContext

def process_data(
    df: pd.DataFrame,           # required param from predecessor
    n_jobs: int = 1,           # optional param with default
    verbose: bool = False,      # optional param with default
    model_type: str = 'basic'   # optional param with default
):
    return (
        f"Processed {df.shape[0]} rows with {n_jobs} jobs, "
        f"verbose={verbose}, "
        f"model_type={model_type}"
    )

@pytest.fixture
def basic_graph():
    """Graph with just data and process nodes using default parameters"""
    graph = ExecutableGraph()
    data_node = Node(
        name="data",
        action_function=lambda: pd.DataFrame(),
        graph=graph
    )
    process_node = Node(
        name="process",
        action_function=process_data,
        graph=graph
    )
    process_node.add_predecessor(data_node, argument_name="df")
    return graph, data_node, process_node

@pytest.fixture
def graph_with_context():
    """Graph with run context overriding some parameters"""
    context = RunContext()
    context.n_jobs = 4
    context.model_type = 'advanced'
    
    graph = ExecutableGraph(run_context=context)
    data_node = Node(
        name="data",
        action_function=lambda: pd.DataFrame(),
        graph=graph
    )
    process_node = Node(
        name="process",
        action_function=process_data,
        graph=graph
    )
    process_node.add_predecessor(data_node, argument_name="df")
    return graph, data_node, process_node

@pytest.fixture
def graph_with_verbose_data():
    """Graph where data node provides verbose parameter"""
    def data_with_config():
        df = pd.DataFrame()
        return df

    context = RunContext()
    context.n_jobs = 4
    context.model_type = 'advanced'
    
    graph = ExecutableGraph(run_context=context)
    data_node = Node(
        name="data",
        action_function=data_with_config,
        graph=graph
    )
    process_node = Node(
        name="process", 
        action_function=process_data,
        graph=graph,
        action_condition_kwarg_map={"verbose": "data"}
    )
    process_node.add_predecessor(data_node, argument_name="df")
    return graph, data_node, process_node

def test_default_parameters(basic_graph):
    """Test that default parameter values are used when not overridden"""
    graph, _, process_node = basic_graph
    graph.execute_graph()
    assert process_node.output == "Processed 0 rows with 1 jobs, verbose=False, model_type=basic"

def test_run_context_override(graph_with_context):
    """Test that run context values override default parameters"""
    graph, _, process_node = graph_with_context
    graph.execute_graph()
    assert process_node.output == "Processed 0 rows with 4 jobs, verbose=False, model_type=advanced"

def test_missing_required_argument():
    """Test that missing required parameters raise appropriate error"""
    def process_data(required_param: str, df: pd.DataFrame):
        return f"Processed {required_param}"
    
    graph = ExecutableGraph()
    
    # Update Node initialization to use named parameters
    data_node = Node(
        name="data",
        action_function=lambda: pd.DataFrame(),
        graph=graph
    )
    
    process_node = Node(
        name="process",
        action_function=process_data,
        graph=graph
    )
    process_node.add_predecessor(data_node, argument_name="df")
    
    with pytest.raises(ValueError) as exc_info:
        graph.execute_graph()
    
    assert "Missing required parameters" in str(exc_info.value)
    assert "required_param" in str(exc_info.value) 