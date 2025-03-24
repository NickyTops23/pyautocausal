# PyAutoCausal

PyAutoCausal is a computational framework for building and executing reproducible causal inference workflows. The library provides a flexible, graph-based system to define, connect, and execute analytical steps in causal analysis pipelines.

## Overview

PyAutoCausal makes causal inference workflows more modular, testable, and reproducible by organizing them as directed graphs of connected computational nodes. This approach simplifies complex analysis pipelines and enables conditional execution paths based on data characteristics.

## Key Features

### Computational Graph Model

- **Node-based architecture**: Each step in your causal analysis is represented as a node
- **Directed workflow**: Nodes are connected to form directed acyclic graphs (DAGs)
- **Automatic execution**: The framework handles execution order and data passing between nodes

### Node Types

- **Standard Nodes**: Process inputs and produce outputs using configurable action functions
- **Input Nodes**: Accept external data and pass it into the workflow
- **Decision Nodes**: Conditionally route execution based on data characteristics or intermediate results

### Data Flow Management

- **Automatic data passing**: Results from predecessor nodes are passed to dependent nodes
- **Type validation**: Optional type checking at connection points
- **Run context**: Global variables available to all nodes

### Output Persistence

- **Multiple output formats**: Save node results in various formats (CSV, Parquet, PNG, JSON, etc.)
- **Automatic type inference**: System tries to determine appropriate format based on output type
- **Configurable paths**: Control where outputs are saved

### Extensibility

- **Library nodes**: Create standardized, reusable nodes for common operations
- **Custom nodes**: Implement specialized node types for specific analytical needs
- **Graph composition**: Merge and connect multiple graphs to build complex workflows
- **Notebook export**: Convert pipelines to Jupyter notebooks for interactive exploration

## Installation

```bash
pip install pyautocausal
```

## Example Usage

### Basic Pipeline

```python
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType
import pandas as pd

def create_sample_data() -> pd.DataFrame:
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 15, 25, 30, 35]
    })

def compute_average(df: pd.DataFrame) -> pd.Series:
    """Compute average values by category"""
    return df.groupby('category')['value'].mean()

# Build graph using builder pattern
graph = (ExecutableGraph(output_path="./outputs")
    .create_node(
        "create_data", 
        create_sample_data,
        save_node=True,
        output_config=OutputConfig(
            output_filename="create_data",
            output_type=OutputType.PARQUET
        )
    )
    .create_node(
        "compute_average",
        compute_average,
        predecessors=["create_data"],
        save_node=True,
        output_config=OutputConfig(
            output_filename="compute_average",
            output_type=OutputType.CSV
        )
    )
)

# Execute the graph
graph.execute_graph()

# Access results
data_node = [n for n in graph.nodes() if n.name == "create_data"][0]
df = data_node.get_result_value()

average_node = [n for n in graph.nodes() if n.name == "compute_average"][0]
averages = average_node.get_result_value()
```

### Conditional Execution

```python
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pathlib import Path

# Define analytical functions
def load_data() -> pd.DataFrame:
    return pd.DataFrame({
        'treatment': [0, 1, 0, 1, 0, 1],
        'outcome': [10, 12, 9, 14, 11, 13],
        'confounder': [5, 6, 4, 7, 5, 8]
    })

def check_sample_size(data: pd.DataFrame) -> bool:
    # Determine if sample size is sufficient for DoubleML
    return len(data) >= 1000

def analyze_with_doubleml(data: pd.DataFrame) -> str:
    # Run DoubleML analysis (simplified)
    return "DoubleML results for small sample"

def analyze_with_ols(data: pd.DataFrame) -> str:
    # Run OLS analysis (simplified)
    outcome = data['outcome']
    treatment = data['treatment']
    confounder = data['confounder']
    
    # Simple OLS model
    import statsmodels.api as sm
    X = sm.add_constant(pd.DataFrame({'treatment': treatment, 'confounder': confounder}))
    model = sm.OLS(outcome, X).fit()
    return model.summary().as_text()

# Create and configure the graph
output_path = Path("./results")
graph = (ExecutableGraph(output_path=output_path)
    .create_input_node("df", input_dtype=pd.DataFrame)
    .create_decision_node(
        "sample_size_check",
        check_sample_size,
        predecessors=["df"],
    )
    .create_node(
        "doubleml",
        analyze_with_doubleml,
        predecessors=["df"],
        save_node=True,
        output_config=OutputConfig(
            output_filename="doubleml_results",
            output_type=OutputType.TEXT 
        )
    )
    .create_node(
        "ols",
        analyze_with_ols,
        predecessors=["df"],
        save_node=True,
        output_config=OutputConfig(
            output_filename="ols_results",
            output_type=OutputType.TEXT 
        )
    )
    .when_true("sample_size_check", "doubleml")
    .when_false("sample_size_check", "ols")
)

# Execute the workflow with input data
data = load_data()
graph.fit(df=data)
```

## Project Structure

```
pyautocausal/
├── orchestration/        # Core graph execution system
│   ├── graph.py          # ExecutableGraph class for workflow orchestration
│   ├── nodes.py          # Node, DecisionNode, InputNode implementations
│   ├── node_state.py     # State management for nodes
│   ├── result.py         # Result wrapper for node outputs
│   ├── base.py           # Base protocols and interfaces
│   └── run_context.py    # Global execution context
├── persistence/          # Output storage and visualization
│   ├── output_config.py  # Configuration for output formats
│   ├── output_types.py   # Supported output types
│   ├── serialization.py  # Serialization utilities
│   ├── type_inference.py # Type-based format selection
│   ├── output_handler.py # Abstract output handler
│   ├── local_output_handler.py # Local filesystem handler
│   ├── notebook_export.py # Export to Jupyter notebooks
│   └── visualizer.py     # Graph visualization tools
├── pipelines/            # Pre-configured analysis pipelines
│   ├── library.py        # Reusable node templates
│   └── example.py        # Example pipelines
└── utils/                # Utility functions
    └── logger.py         # Logging configuration
```

## Benefits

- **Reproducibility**: Explicitly defined workflows with saved outputs
- **Modularity**: Break complex analyses into independent, reusable components
- **Testability**: Test individual components or entire workflows
- **Flexibility**: Adapt execution paths based on data characteristics
- **Transparency**: Clearly visible analysis steps and data flow

## Contributing

Contributions to PyAutoCausal are welcome! Please see our contributing guidelines for more information.

## License

[Add your license information here]

## Contact

[Add your contact information here]
