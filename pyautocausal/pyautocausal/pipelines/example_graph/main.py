"""Main execution and demonstration for the causal inference pipeline.

This module contains the main execution function that demonstrates the full workflow:
1. Generate mock data with staggered treatment
2. Execute the causal inference graph  
3. Export results and visualizations
"""

from pathlib import Path
import pandas as pd

from pyautocausal.orchestration.graph import ExecutableGraph, RunContext
from . import create_panel_graph, create_cross_sectional_graph
from ..library.conditions import has_multiple_periods
from .utils import setup_output_directories


def main(file_path: str | Path, output_dir: str | Path):
    """
    Load data, determine the appropriate analysis path (panel or cross-sectional),
    and execute the corresponding causal inference graph.

    Args:
        file_path (str or Path): Path to the input data file (CSV or Parquet).
        output_dir (str or Path): Path to the output directory.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)

    # 1. Load Data
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Parquet.")
        
    # 2. Setup Output Directories
    abs_text_dir, abs_plot_dir, abs_data_dir = setup_output_directories(output_dir)

    # 3. Determine Data Type and Create Appropriate Graph
    if has_multiple_periods(df):
        print("Panel data detected. Creating and executing the panel data graph.")
        graph = create_panel_graph(abs_text_dir, abs_plot_dir)
    else:
        print("Cross-sectional data detected. Creating and executing the cross-sectional data graph.")
        graph = create_cross_sectional_graph(abs_text_dir, abs_plot_dir)
    
    # 4. Execute Graph
    graph.execute(
        input_data={"df": df}, 
        output_dir=output_dir, 
        run_context=RunContext(
            y_col="y",
            id_col="id_unit",
            t_col="t",
            treat_col="treat"
        )
    )
    
    print(f"\nGraph execution complete. Outputs are saved in: {output_dir}")


if __name__ == "__main__":
    # This is placeholder for if you want to make this script runnable
    # For example:
    # if len(sys.argv) > 2:
    #     main(sys.argv[1], sys.argv[2])
    # else:
    #     print("Usage: python -m pyautocausal.pipelines.example_graph.main <path_to_data> <output_dir>")
    pass 