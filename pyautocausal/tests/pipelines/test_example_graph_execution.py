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

from pyautocausal.pipelines.example_graph import causal_pipeline, simple_graph
from pyautocausal.pipelines.mock_data import generate_mock_data


class TestExampleGraphExecution:
    """Test suite for actual execution of the refactored example graph."""

    def test_simple_graph_execution(self):
        """Test that the simple graph executes successfully with cross-sectional data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate single period data (cross-sectional)
            data = generate_mock_data(n_units=50, n_periods=1, n_treated=25, staggered_treatment=False)
            
            # Create and execute simple graph
            graph = simple_graph()
            result = graph.fit(df=data)
            
            # Check that execution completed
            assert result is not None
            
            # Verify expected nodes completed
            completed_nodes = []
            failed_nodes = []
            
            for node in graph.nodes():
                if hasattr(node, 'state') and hasattr(node.state, 'name'):
                    if node.state.name == 'COMPLETED':
                        completed_nodes.append(node.name)
                    elif node.state.name == 'FAILED':
                        failed_nodes.append(node.name)
            
            # Should have no failed nodes
            assert len(failed_nodes) == 0, f"Failed nodes: {failed_nodes}"
            
            # Should have completed cross-sectional branch
            expected_completed = ['df', 'multi_period', 'stand_spec', 'ols_stand', 'ols_stand_output']
            for expected_node in expected_completed:
                assert expected_node in completed_nodes, f"Expected {expected_node} to complete"
            
            print(f"✓ Simple graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_cross_sectional_branch_execution(self):
        """Test cross-sectional analysis branch with single period data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate single period data to trigger cross-sectional analysis
            data = generate_mock_data(n_units=100, n_periods=1, n_treated=50, staggered_treatment=False)
            
            # Create and execute full pipeline
            graph = causal_pipeline(output_path)
            result = graph.fit(df=data)
            
            # Check execution results
            completed_nodes = []
            failed_nodes = []
            
            for node in graph.nodes():
                if hasattr(node, 'state') and hasattr(node.state, 'name'):
                    if node.state.name == 'COMPLETED':
                        completed_nodes.append(node.name)
                    elif node.state.name == 'FAILED':
                        failed_nodes.append(node.name)
            
            assert len(failed_nodes) == 0, f"Failed nodes: {failed_nodes}"
            
            # Should execute cross-sectional branch
            cross_sectional_nodes = ['stand_spec', 'ols_stand', 'ols_stand_output']
            for node in cross_sectional_nodes:
                assert node in completed_nodes, f"Cross-sectional node {node} should have completed"
            
            # Should NOT execute panel-specific nodes
            panel_nodes = ['synthdid_spec', 'did_spec', 'event_spec', 'stag_spec']
            for node in panel_nodes:
                assert node not in completed_nodes, f"Panel node {node} should not have executed with single period data"
            
            # Check output files were created
            assert (output_path / "text").exists()
            assert (output_path / "plots").exists()
            assert (output_path / "notebooks").exists()
            
            # Check that basic cleaning metadata was created
            basic_metadata_file = output_path / "text" / "basic_cleaning_metadata.txt"
            assert basic_metadata_file.exists(), "Basic cleaning metadata should be created"
            
            print(f"✓ Cross-sectional branch executed successfully: {len(completed_nodes)} nodes completed")

    def test_panel_data_execution(self):
        """Test panel data analysis with multiple periods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate panel data (multiple periods, no staggered treatment)
            data = generate_mock_data(n_units=50, n_periods=3, n_treated=25, staggered_treatment=False)
            
            # Create and execute full pipeline
            graph = causal_pipeline(output_path)
            result = graph.fit(df=data)
            
            # Check execution results
            completed_nodes = []
            failed_nodes = []
            
            for node in graph.nodes():
                if hasattr(node, 'state') and hasattr(node.state, 'name'):
                    if node.state.name == 'COMPLETED':
                        completed_nodes.append(node.name)
                    elif node.state.name == 'FAILED':
                        failed_nodes.append(node.name)
            
            assert len(failed_nodes) == 0, f"Failed nodes: {failed_nodes}"
            
            # Should execute panel branch
            panel_nodes = ['panel_cleaned_data', 'panel_cleaning_metadata']
            for node in panel_nodes:
                assert node in completed_nodes, f"Panel node {node} should have completed"
            
            # Should NOT execute cross-sectional branch
            assert 'cross_sectional_cleaned_data' not in completed_nodes, "Cross-sectional cleaning should not execute with panel data"
            
            # Check that panel cleaning metadata was created
            panel_metadata_file = output_path / "text" / "panel_cleaning_metadata.txt"
            assert panel_metadata_file.exists(), "Panel cleaning metadata should be created"
            
            print(f"✓ Panel data branch executed successfully: {len(completed_nodes)} nodes completed")

    def test_staggered_treatment_execution(self):
        """Test staggered treatment analysis (most complex branch)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate staggered treatment data
            data = generate_mock_data(n_units=100, n_periods=5, n_treated=60, staggered_treatment=True)
            
            # Save data for the pipeline
            data_csv_path = output_path / "notebooks" / "causal_pipeline_data.csv"
            data_csv_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(data_csv_path, index=False)
            
            # Create and execute full pipeline
            graph = causal_pipeline(output_path)
            result = graph.fit(df=data)
            
            # Check execution results
            completed_nodes = []
            failed_nodes = []
            
            for node in graph.nodes():
                if hasattr(node, 'state') and hasattr(node.state, 'name'):
                    if node.state.name == 'COMPLETED':
                        completed_nodes.append(node.name)
                    elif node.state.name == 'FAILED':
                        failed_nodes.append(node.name)
                    elif node.state.name == 'PASSED':
                        # PASSED means skipped due to decision branching - this is normal
                        pass
            
            assert len(failed_nodes) == 0, f"Failed nodes: {failed_nodes}"
            
            # Should execute staggered DiD branch
            staggered_nodes = ['stag_spec', 'ols_stag']
            for node in staggered_nodes:
                assert node in completed_nodes, f"Staggered DiD node {node} should have completed"
            
            # Should execute one of the Callaway & Sant'Anna methods
            cs_nodes = ['cs_never_treated', 'cs_not_yet_treated']
            cs_executed = any(node in completed_nodes for node in cs_nodes)
            assert cs_executed, "At least one Callaway & Sant'Anna method should have executed"
            
            print(f"✓ Staggered treatment analysis executed successfully: {len(completed_nodes)} nodes completed")

    def test_output_file_creation(self):
        """Test that all expected output files are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate data and run pipeline
            data = generate_mock_data(n_units=50, n_periods=3, n_treated=25, staggered_treatment=False)
            
            # Save data for export functions
            data_csv_path = output_path / "notebooks" / "causal_pipeline_data.csv"
            data_csv_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(data_csv_path, index=False)
            
            graph = causal_pipeline(output_path)
            result = graph.fit(df=data)
            
            # Export outputs using the pipeline's export function
            from pyautocausal.pipelines.example_graph.utils import export_outputs
            export_outputs(graph, output_path)
            
            # Check directory structure
            assert (output_path / "plots").exists(), "Plots directory should exist"
            assert (output_path / "text").exists(), "Text directory should exist"
            assert (output_path / "notebooks").exists(), "Notebooks directory should exist"
            
            # Check visualization file
            viz_file = output_path / "text" / "causal_pipeline_visualization.md"
            assert viz_file.exists(), "Graph visualization should be created"
            assert viz_file.stat().st_size > 0, "Visualization file should not be empty"
            
            # Check notebook file
            notebook_file = output_path / "notebooks" / "causal_pipeline_execution.ipynb"
            assert notebook_file.exists(), "Jupyter notebook should be created"
            
            # Verify notebook content
            with open(notebook_file, 'r') as f:
                notebook_content = json.load(f)
            assert 'cells' in notebook_content, "Notebook should have cells"
            assert len(notebook_content['cells']) > 0, "Notebook should have at least one cell"
            
            # Check metadata files
            basic_metadata = output_path / "text" / "basic_cleaning_metadata.txt"
            assert basic_metadata.exists(), "Basic cleaning metadata should exist"
            
            print("✓ All expected output files created successfully")

    def test_node_state_consistency(self):
        """Test that node states are consistent and logical."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate cross-sectional data
            data = generate_mock_data(n_units=50, n_periods=1, n_treated=25, staggered_treatment=False)
            
            graph = causal_pipeline(output_path)
            result = graph.fit(df=data)
            
            # Analyze node states
            completed_nodes = []
            passed_nodes = []  # Skipped due to decision branching
            failed_nodes = []
            
            for node in graph.nodes():
                if hasattr(node, 'state') and hasattr(node.state, 'name'):
                    if node.state.name == 'COMPLETED':
                        completed_nodes.append(node.name)
                    elif node.state.name == 'PASSED':
                        passed_nodes.append(node.name)
                    elif node.state.name == 'FAILED':
                        failed_nodes.append(node.name)
            
            # No nodes should fail
            assert len(failed_nodes) == 0, f"Failed nodes: {failed_nodes}"
            
            # Should have reasonable number of completed nodes
            assert len(completed_nodes) >= 5, f"Expected at least 5 completed nodes, got {len(completed_nodes)}"
            
            # Should have some passed (skipped) nodes due to decision branching
            assert len(passed_nodes) > 0, f"Expected some passed nodes due to branching, got {len(passed_nodes)}"
            
            # Cross-sectional data should skip panel-specific nodes
            panel_specific_nodes = ['synthdid_spec', 'did_spec', 'event_spec', 'stag_spec']
            for node in panel_specific_nodes:
                assert node in passed_nodes, f"Panel node {node} should be skipped with cross-sectional data"
            
            total_nodes = len(completed_nodes) + len(passed_nodes) + len(failed_nodes)
            print(f"✓ Node states consistent: {len(completed_nodes)} completed, {len(passed_nodes)} skipped, {len(failed_nodes)} failed (total: {total_nodes})")


if __name__ == "__main__":
    # Run tests directly
    test_suite = TestExampleGraphExecution()
    
    print("Running example graph execution tests...")
    
    try:
        test_suite.test_simple_graph_execution()
        print("✓ Simple graph execution test passed")
        
        test_suite.test_cross_sectional_branch_execution()
        print("✓ Cross-sectional branch execution test passed")
        
        test_suite.test_panel_data_execution()
        print("✓ Panel data execution test passed")
        
        test_suite.test_output_file_creation()
        print("✓ Output file creation test passed")
        
        test_suite.test_node_state_consistency()
        print("✓ Node state consistency test passed")
        
        print("\n✓ All execution tests passed! The refactored example graph runs correctly and creates expected outputs.")
        
    except Exception as e:
        print(f"\n✗ Execution test failed: {e}")
        raise 