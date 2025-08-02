"""Execution tests for the refactored example graph.

This test module verifies that the refactored example graph:
1. Actually executes successfully end-to-end
2. Creates the expected output files
3. Routes data through different analysis branches correctly
4. Produces valid results
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path

from pyautocausal.pipelines.example_graph import create_cross_sectional_graph, create_panel_graph
from pyautocausal.pipelines.mock_data import generate_mock_data


class TestExampleGraphExecution:
    """Test suite for actual execution of the refactored example graph."""

    def test_cross_sectional_graph_execution(self):
        """Test the cross-sectional graph with single period data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            abs_text_dir = output_path / "text"
            abs_plot_dir = output_path / "plots"
            
            # Generate single period data
            data = generate_mock_data(n_units=100, n_periods=1, n_treated=50, staggered_treatment=False)
            
            # Create and execute cross-sectional graph
            graph = create_cross_sectional_graph(abs_text_dir, abs_plot_dir)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for the cross-sectional path
            expected_nodes = {'df', 'basic_cleaning', 'multi_period', 'cross_sectional_cleaned_data', 'ate_spec', 'ate_estimate', 'cate_spec', 'cate_estimate'}
            assert expected_nodes.issubset(completed_nodes)
            
            # Panel nodes should not be in the graph at all
            panel_nodes = {'panel_cleaned_data', 'synthdid_spec', 'did_spec', 'event_spec', 'stag_spec'}
            graph_nodes = {node.name for node in graph.nodes()}
            assert not panel_nodes.intersection(graph_nodes)
            
            print(f"✓ Cross-sectional graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_panel_graph_execution(self):
        """Test the panel graph with multi-period data and no staggered treatment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            abs_text_dir = output_path / "text"
            abs_plot_dir = output_path / "plots"
            
            # Generate panel data
            data = generate_mock_data(n_units=50, n_periods=3, n_treated=25, staggered_treatment=False)
            
            # Create and execute panel graph
            graph = create_panel_graph(abs_text_dir, abs_plot_dir)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for a simple panel data path
            expected_nodes = {'df', 'basic_cleaning', 'multi_period', 'panel_cleaned_data', 'event_spec', 'event_estimate'}
            assert expected_nodes.issubset(completed_nodes)
            
            # Cross-sectional nodes should not be in the graph
            cross_sectional_nodes = {'cross_sectional_cleaned_data', 'ate_spec', 'cate_spec'}
            graph_nodes = {node.name for node in graph.nodes()}
            assert not cross_sectional_nodes.intersection(graph_nodes)
            
            print(f"✓ Panel graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_staggered_treatment_in_panel_graph(self):
        """Test the staggered treatment path within the panel graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            abs_text_dir = output_path / "text"
            abs_plot_dir = output_path / "plots"
            
            # Generate staggered treatment data
            data = generate_mock_data(n_units=100, n_periods=5, n_treated=60, staggered_treatment=True)
            
            # Create and execute panel graph
            graph = create_panel_graph(abs_text_dir, abs_plot_dir)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}

            assert not failed_nodes, f"Failed nodes: {failed_nodes}"

            # Expected nodes for the staggered treatment path
            expected_nodes = {'df', 'basic_cleaning', 'multi_period', 'panel_cleaned_data', 'stag_spec'}
            assert expected_nodes.issubset(completed_nodes)

            # One of the Callaway & Sant'Anna branches should execute
            cs_executed = 'cs_never_treated' in completed_nodes or 'cs_not_yet_treated' in completed_nodes
            assert cs_executed, "A Callaway & Sant'Anna branch should have executed."

            print(f"✓ Staggered treatment path in panel graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_output_file_creation_for_panel_graph(self):
        """Test that the panel graph creates all expected output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            abs_text_dir = output_path / "text"
            abs_plot_dir = output_path / "plots"
            
            # Ensure output directories are created before running the graph
            abs_text_dir.mkdir(parents=True, exist_ok=True)
            abs_plot_dir.mkdir(parents=True, exist_ok=True)
            (output_path / "notebooks").mkdir(parents=True, exist_ok=True)

            data = generate_mock_data(n_units=50, n_periods=3, n_treated=25, staggered_treatment=False)
            
            # Create and execute panel graph
            graph = create_panel_graph(abs_text_dir, abs_plot_dir)
            graph.fit(df=data)
            
            from pyautocausal.persistence.notebook_export import NotebookExporter
            exporter = NotebookExporter(graph)
            exporter.export_notebook(str(output_path / "notebooks" / "panel_execution.ipynb"))
            
            assert (output_path / "text" / "panel_cleaning_metadata.txt").exists()
            # Check that the plots directory is not empty, rather than checking for a specific file
            assert any(abs_plot_dir.iterdir())
            assert (output_path / "notebooks" / "panel_execution.ipynb").exists()
            
            print("✓ Panel graph created all expected output files.")

    def test_node_states_in_cross_sectional_graph(self):
        """Test that node states are consistent in the cross-sectional graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            abs_text_dir = output_path / "text"
            abs_plot_dir = output_path / "plots"
            
            data = generate_mock_data(n_units=50, n_periods=1, n_treated=25, staggered_treatment=False)
            
            graph = create_cross_sectional_graph(abs_text_dir, abs_plot_dir)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            passed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'PASSED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}

            assert not failed_nodes
            assert len(completed_nodes) > 5
            # In a single-path graph, we might not have 'PASSED' nodes unless there are internal decisions
            # For the cross-sectional graph, we expect most nodes to complete.
            
            print(f"✓ Cross-sectional graph node states consistent: {len(completed_nodes)} completed, {len(passed_nodes)} skipped.")


if __name__ == "__main__":
    # Run tests directly
    test_suite = TestExampleGraphExecution()
    
    print("Running example graph execution tests...")
    
    try:
        test_suite.test_cross_sectional_graph_execution()
        print("✓ Cross-sectional graph execution test passed")
        
        test_suite.test_panel_graph_execution()
        print("✓ Panel graph execution test passed")
        
        test_suite.test_staggered_treatment_in_panel_graph()
        print("✓ Staggered treatment path in panel graph test passed")

        test_suite.test_output_file_creation_for_panel_graph()
        print("✓ Panel graph output file creation test passed")
        
        test_suite.test_node_states_in_cross_sectional_graph()
        print("✓ Cross-sectional graph node state consistency test passed")
        
        print("\n✓ All execution tests passed! The refactored example graph runs correctly and creates expected outputs.")
        
    except Exception as e:
        print(f"\n✗ Execution test failed: {e}")
        raise 