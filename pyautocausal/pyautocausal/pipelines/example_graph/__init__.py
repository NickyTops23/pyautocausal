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
# All branch imports are now done locally within each function as needed
from .core import (
    _create_shared_head,
    create_panel_decision_structure,
    configure_panel_decision_paths,
    create_cross_sectional_decision_structure,
    configure_cross_sectional_decision_paths
)


def create_panel_graph(abs_text_dir: str, abs_plot_dir: str) -> ExecutableGraph:
    """Create the panel data causal inference graph."""
    graph = ExecutableGraph()
    
    # 1. Shared Head
    _create_shared_head(graph)
    
    # 2. Panel Decision Structure
    create_panel_decision_structure(graph, abs_text_dir=abs_text_dir)
    
    # 3. Add Analysis Branches (using complete functions with saving functionality)
    from .branches import (
        create_did_branch,
        create_event_study_branch, 
        create_synthetic_did_branch,
        create_staggered_did_branch
    )
    
    create_did_branch(graph, Path(abs_text_dir))
    create_event_study_branch(graph, Path(abs_text_dir), Path(abs_plot_dir))
    create_synthetic_did_branch(graph, Path(abs_plot_dir))
    create_staggered_did_branch(graph, Path(abs_text_dir), Path(abs_plot_dir))
    
    # 4. Configure Paths
    configure_panel_decision_paths(graph)
    
    return graph


def create_cross_sectional_graph(abs_text_dir: str, abs_plot_dir: str) -> ExecutableGraph:
    """Create the cross-sectional data causal inference graph."""
    graph = ExecutableGraph()
    
    # 1. Shared Head
    _create_shared_head(graph)
    
    # 2. Cross-Sectional Decision Structure
    create_cross_sectional_decision_structure(graph, abs_text_dir=abs_text_dir)
    
    # 3. Add Analysis Branches (using complete function with saving functionality)
    from .branches import create_cross_sectional_branch
    create_cross_sectional_branch(graph, Path(abs_text_dir))
    
    # 4. Configure Paths
    configure_cross_sectional_decision_paths(graph)
    
    return graph


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
    if not isinstance(output_path, Path):
        output_path = Path(output_path);
    # Initialize graph and directories
    graph = ExecutableGraph()
    graph.configure_runtime(output_path=output_path)
    abs_plots_dir, abs_text_dir, abs_notebooks_dir = setup_output_directories(output_path)
    
    # Create the complete pipeline with both panel and cross-sectional paths
    _create_shared_head(graph)
    
    # Add top-level routing decision based on data characteristics
    from .core import has_multiple_periods
    graph.create_decision_node(
        'is_panel_data',
        condition=has_multiple_periods.get_function(),
        predecessors=["basic_cleaning"]
    )
    
    # Create both decision structures with routing dependencies
    create_panel_decision_structure(graph, abs_text_dir, predecessor="is_panel_data")
    create_cross_sectional_decision_structure(graph, abs_text_dir, predecessor="is_panel_data")
    
    # Add all analysis branches (using complete functions with saving functionality)
    from .branches import (
        create_did_branch,
        create_event_study_branch, 
        create_synthetic_did_branch,
        create_staggered_did_branch,
        create_cross_sectional_branch
    )
    
    create_did_branch(graph, abs_text_dir)
    create_event_study_branch(graph, abs_text_dir, abs_plots_dir)
    create_synthetic_did_branch(graph, abs_plots_dir)
    create_staggered_did_branch(graph, abs_text_dir, abs_plots_dir)
    create_cross_sectional_branch(graph, abs_text_dir)
    
    # Configure top-level routing
    graph.when_true("is_panel_data", "panel_cleaned_data")
    graph.when_false("is_panel_data", "cross_sectional_cleaned_data")
    
    # Configure both decision paths
    configure_panel_decision_paths(graph)
    configure_cross_sectional_decision_paths(graph)
    
    return graph


def simple_graph() -> ExecutableGraph:
    """Create a simple graph for testing purposes.
    
    This creates a basic pipeline with cross-sectional and DiD branches
    for testing serialization and basic functionality.
    
    Returns:
        Configured ExecutableGraph ready for execution
    """
    from pathlib import Path
    
    graph = ExecutableGraph()
    
    # Create the shared head nodes
    _create_shared_head(graph)
    
    # Add simple cross-sectional branches for testing
    create_cross_sectional_decision_structure(graph, abs_text_dir=Path("/tmp/text"))
    from .branches import create_cross_sectional_branch
    create_cross_sectional_branch(graph, Path("/tmp/text"))
    configure_cross_sectional_decision_paths(graph)
    
    return graph


# Import main execution function and utilities for convenience
from .main import main
from .utils import export_outputs, setup_output_directories

# Backward compatibility alias for the test
_export_outputs = export_outputs

__all__ = [
    'causal_pipeline',
    'simple_graph', 
    'main',
    'export_outputs',
    '_export_outputs'
] 