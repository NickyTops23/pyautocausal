"""PyAutoCausal Example Graph

A comprehensive causal inference pipeline that automatically selects appropriate 
methods based on data characteristics.

The pipeline supports:
- Cross-sectional analysis (single period)
- Synthetic DiD (single treated unit, multiple periods)  
- Standard DiD (multiple periods, insufficient for staggered)
- Event study (multiple periods, non-staggered treatment)
- Staggered DiD with Callaway & Sant'Anna methods

## Quick Start

```python
from pyautocausal.pipelines.example_graph import causal_pipeline
from pathlib import Path
import pandas as pd

# Create the pipeline
output_path = Path('output')
graph = causal_pipeline(output_path)

# Run with your data
result = graph.fit(df=your_dataframe)
```

## Architecture

The pipeline is organized into separate modules:

- `core.py`: Core decision structure and routing logic
- `branches.py`: Specific causal analysis method implementations  
- `utils.py`: Helper functions for setup and output
- `main.py`: Main execution and demo functionality
"""

from pathlib import Path
from pyautocausal.orchestration.graph import ExecutableGraph
from .core import create_core_decision_structure, configure_core_decision_paths
from .branches import create_all_analysis_branches
from .utils import setup_output_directories, create_simple_graph


def causal_pipeline(output_path: Path) -> ExecutableGraph:
    """Create a comprehensive causal inference pipeline that automatically selects
    appropriate methods based on data characteristics.
    
    The pipeline supports:
    - Cross-sectional analysis (single period)
    - Synthetic DiD (single treated unit, multiple periods)
    - Standard DiD (multiple periods, insufficient for staggered)
    - Event study (multiple periods, non-staggered treatment)
    - Staggered DiD with Callaway & Sant'Anna methods
    
    Args:
        output_path: Directory where results will be saved
        
    Returns:
        Configured ExecutableGraph ready for execution
    """
    # Initialize graph and directories
    graph = ExecutableGraph()
    graph.configure_runtime(output_path=output_path)
    abs_plots_dir, abs_text_dir, abs_notebooks_dir = setup_output_directories(output_path)
    
    # Create the complete pipeline
    create_core_decision_structure(graph, abs_text_dir)
    create_all_analysis_branches(graph, abs_text_dir, abs_plots_dir)
    configure_core_decision_paths(graph)
    
    return graph


def simple_graph() -> ExecutableGraph:
    """Create a simple graph for testing purposes.
    
    This creates a basic pipeline with cross-sectional and DiD branches
    for testing serialization and basic functionality.
    
    Returns:
        Configured ExecutableGraph ready for execution
    """
    return create_simple_graph()


# Import main execution function and utilities for convenience
from .main import main
from .utils import export_outputs

# Backward compatibility alias for the test
_export_outputs = export_outputs

__all__ = [
    'causal_pipeline',
    'simple_graph', 
    'main',
    'export_outputs',
    '_export_outputs'
] 