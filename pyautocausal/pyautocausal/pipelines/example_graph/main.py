"""Main execution and demonstration for the causal inference pipeline.

This module contains the main execution function that demonstrates the full workflow:
1. Generate mock data with staggered treatment
2. Execute the causal inference graph  
3. Export results and visualizations
"""

from pathlib import Path

from pyautocausal.pipelines.mock_data import generate_mock_data
from .core import create_core_decision_structure, configure_core_decision_paths
from .branches import create_all_analysis_branches  
from .utils import setup_output_directories, print_execution_summary, export_outputs, print_data_characteristics


def main():
    """Main execution function for running the example graph.
    
    This demonstrates the full workflow:
    1. Generate mock data with staggered treatment
    2. Execute the causal inference graph
    3. Export results and visualizations
    """
    # Setup
    output_path = Path('output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("======= Running PyAutoCausal Example Graph =======")
    
    # Initialize graph and create directory structure
    from pyautocausal.orchestration.graph import ExecutableGraph
    graph = ExecutableGraph()
    graph.configure_runtime(output_path=output_path)
    abs_plots_dir, abs_text_dir, abs_notebooks_dir = setup_output_directories(output_path)
    
    # Build the complete graph
    print("Building causal inference graph...")
    create_core_decision_structure(graph, abs_text_dir)
    create_all_analysis_branches(graph, abs_text_dir, abs_plots_dir)
    configure_core_decision_paths(graph)
    print(f"Graph built with {len(list(graph.nodes()))} nodes")
    
    # Generate realistic mock data for demonstration
    data = generate_mock_data(
        n_units=100, 
        n_periods=10, 
        n_treated=70, 
        staggered_treatment=True, 
        noise_to_signal_ratio=1.5
    )
    
    # Save data for reference
    data_csv_path = output_path / "notebooks" / "causal_pipeline_data.csv"
    data.to_csv(data_csv_path, index=False)
    print(f"Mock data generated and saved to {data_csv_path}")

    # Display data characteristics
    print("\n" + "="*30 + " DATA CHARACTERISTICS " + "="*30)
    print_data_characteristics(data)
    print("\n")

    # Execute graph
    try:
        print("Executing causal inference graph...")
        graph.fit(df=data)
        print("Graph execution completed successfully.")
        
    except Exception as e:
        print(f"Error during graph execution: {e}")
        return

    # Results summary and export
    print("\n" + "="*30 + " EXECUTION SUMMARY " + "="*30)
    print_execution_summary(graph)
    print("-" * 50)
    
    export_outputs(graph, output_path)
    print("\n======= Example Graph Run Finished =======")


if __name__ == "__main__":
    main() 