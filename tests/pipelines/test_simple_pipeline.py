import pytest
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.local_output_handler import LocalOutputHandler


def create_sample_data() -> pd.DataFrame:
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 15, 25, 30, 35]
    })

def compute_average(df: pd.DataFrame) -> pd.Series:
    """Compute average values by category"""
    return df.groupby('category')['value'].mean()

def create_plot(avg_data: pd.Series) -> bytes:
    """Create a plot visualization of the averages"""
    plt.figure(figsize=(8, 6))
    avg_data.plot(kind='bar')
    plt.xlabel('Category')
    plt.ylabel('Average Value')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    return buffer.getvalue()

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing"""
    return create_sample_data()

@pytest.fixture
def pipeline_graph(tmp_path):
    """Create a configured pipeline graph with all nodes"""
    graph = ExecutableGraph(
        output_handler=LocalOutputHandler(tmp_path / 'outputs')
    )
    
    # Data node
    data_node = Node(
        "create_data", 
        graph,
        create_sample_data,
        save_node=True
    )
    
    # Average computation node
    average_node = Node(
        "compute_average",
        graph,
        compute_average,
        save_node=True
    )
    average_node.add_predecessor(data_node, argument_name="df")
    
    # Plot creation node
    plot_node = Node(
        "create_plot",
        graph,
        create_plot,
        save_node=True
    )
    plot_node.add_predecessor(average_node, argument_name="avg_data")
    
    return graph, data_node, average_node, plot_node

@pytest.fixture
def executed_pipeline(pipeline_graph):
    """Execute the pipeline and return the graph and nodes"""
    graph, data_node, average_node, plot_node = pipeline_graph
    graph.execute_graph()
    return graph, data_node, average_node, plot_node

def test_pipeline_execution(executed_pipeline):
    """Test that all nodes complete execution"""
    _, data_node, average_node, plot_node = executed_pipeline
    
    assert data_node.is_completed()
    assert average_node.is_completed()
    assert plot_node.is_completed()

def test_data_node_output(executed_pipeline):
    """Test the output of the data node"""
    _, data_node, _, _ = executed_pipeline
    
    df = data_node.output
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['category', 'value']
    assert len(df) == 6

def test_average_node_output(executed_pipeline):
    """Test the output of the average computation node"""
    _, _, average_node, _ = executed_pipeline
    
    averages = average_node.output
    assert isinstance(averages, pd.Series)
    assert len(averages) == 2
    assert all(averages.index == ['A', 'B'])
    assert all(averages > 0)

def test_plot_node_output(executed_pipeline):
    """Test the output of the plot creation node"""
    _, _, _, plot_node = executed_pipeline
    
    plot = plot_node.output
    assert isinstance(plot, bytes)
    assert len(plot) > 0

def test_output_files_creation(executed_pipeline, tmp_path):
    """Test that all output files are created correctly"""
    output_dir = tmp_path / 'outputs'
    assert output_dir.exists()
    
    expected_files = [
        'create_data.parquet',
        'compute_average.parquet',
        'create_plot.png'
    ]
    
    for filename in expected_files:
        file_path = output_dir / filename
        assert file_path.exists(), f"Expected output file {filename} not found"
        assert file_path.stat().st_size > 0, f"Output file {filename} is empty"

def test_output_files_content(executed_pipeline, tmp_path, sample_data):
    """Test the content of output files"""
    output_dir = tmp_path / 'outputs'
    
    # Test parquet file
    df = pd.read_parquet(output_dir / 'create_data.parquet')
    pd.testing.assert_frame_equal(df, sample_data)
    
    # Test CSV file
    averages = pd.read_csv(output_dir / 'compute_average.csv', index_col=0)
    assert len(averages) == 2
    assert all(averages.index == ['A', 'B'])
    
    # Test PNG file
    with open(output_dir / 'create_plot.png', 'rb') as f:
        plot_data = f.read()
    assert len(plot_data) > 0

