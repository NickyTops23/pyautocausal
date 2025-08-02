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
import os

from pyautocausal.pipelines.example_graph import create_cross_sectional_graph, create_panel_graph
from pyautocausal.pipelines.mock_data import generate_mock_data


class TestExampleGraphExecution:
    """Test suite for actual execution of the refactored example graph."""

    def test_cross_sectional_graph_execution(self):
        """Test the cross-sectional graph with single period data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate single period data
            data = generate_mock_data(n_units=100, n_periods=1, n_treated=50, staggered_treatment=False)
            
            # Create and execute cross-sectional graph
            graph = create_cross_sectional_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for the cross-sectional path
            expected_nodes = {'cross_sectional_cleaned_data', 'basic_cleaning', 'df', 'ols_stand', 'ols_stand_output'}
            assert expected_nodes.issubset(completed_nodes)
            
            print(f"✓ Cross-sectional graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_panel_graph_execution(self):
        """Test the panel graph with multi-period data and no staggered treatment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate panel data
            data = generate_mock_data(n_units=50, n_periods=3, n_treated=25, staggered_treatment=False)
            
            # Create and execute panel graph
            graph = create_panel_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for a simple panel data path
            expected_nodes = {'save_event_output', 'stag_treat', 'ols_event', 'panel_cleaned_data', 'multi_post_periods', 'single_treated_unit', 'df', 'basic_cleaning', 'event_spec', 'event_plot'}
            assert expected_nodes.issubset(completed_nodes)

            # check that the results were saved
            assert set(os.listdir(output_path)) == {'save_event_output.txt', 'event_study_plot.png'}
            
            print(f"✓ Panel graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_staggered_treatment_in_panel_graph(self):
        """Test the staggered treatment path within the panel graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate staggered treatment data
            data = generate_mock_data(n_units=100, n_periods=5, n_treated=60, staggered_treatment=True)
            
            # Create and execute panel graph
            graph = create_panel_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}

            assert not failed_nodes, f"Failed nodes: {failed_nodes}"

            # Expected nodes for the staggered treatment path
            expected_nodes = {'has_never_treated', 'panel_cleaned_data', 'single_treated_unit', 'basic_cleaning', 'save_cs_never_treated', 'cs_never_treated', 'ols_stag', 'save_stag_output', 'stag_spec', 'df', 'stag_treat', 'multi_post_periods', 'stag_event_plot', 'cs_never_treated_plot'}
            assert expected_nodes.issubset(completed_nodes)
            
            # export graph to notebook
            from pyautocausal.persistence.notebook_export import NotebookExporter
            exporter = NotebookExporter(graph)
            exporter.export_notebook(str(output_path  / "panel_execution.ipynb"))
            
            assert (output_path / "panel_execution.ipynb").exists()
            
            print(f"✓ Staggered treatment path in panel graph executed successfully: {len(completed_nodes)} nodes completed")


    def test_synthetic_did_branch(self):
        """Test Synthetic DiD branch (multiple periods, single treated unit)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
    
        # Create data with single treated unit (triggers Synthetic DiD branch)
        data = generate_mock_data(n_units=50, n_periods=5, n_treated=1)
        
        graph = create_panel_graph(output_path)
        graph.fit(df=data)
        
        
        
        # Verify nodes completed
        completed_nodes = [node.name for node in graph.nodes() 
                        if hasattr(node, 'state') and node.state.name == 'COMPLETED']
        failed_nodes = [node.name for node in graph.nodes() 
                    if hasattr(node, 'state') and node.state.name == 'FAILED']
        
        # Ensure no nodes failed
        assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
        
        assert set(os.listdir(output_path)) == {'synthdid_plot.png'}


    def test_node_states_in_cross_sectional_graph(self):
        """Test that node states are consistent in the cross-sectional graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            data = generate_mock_data(n_units=50, n_periods=1, n_treated=25, staggered_treatment=False)
            
            graph = create_cross_sectional_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            passed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'PASSED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}

            assert not failed_nodes
            assert len(completed_nodes) > 5
            # In a single-path graph, we might not have 'PASSED' nodes unless there are internal decisions
            # For the cross-sectional graph, we expect most nodes to complete.
            
            print(f"✓ Cross-sectional graph node states consistent: {len(completed_nodes)} completed, {len(passed_nodes)} skipped.")

    def test_standard_did_branch(self):
        """Test standard DiD branch (multiple periods, multiple treated units, insufficient post periods)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
        
        # Create data with 2 periods only (insufficient for staggered treatment)
        data = generate_mock_data(n_units=100, n_periods=2, n_treated=50)
        
        graph = create_panel_graph(output_path)
        graph.fit(df=data)
        
        # Check that standard DiD files are generated
        text_files = list((output_path).glob("*.txt"))
        expected_files = ["save_ols_did.txt"]
        
        for expected_file in expected_files:
            assert any(expected_file in str(f) for f in text_files), f"Expected file {expected_file} not found"
        
        # Verify nodes completed
        completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
        failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
        
        assert completed_nodes == {'did_spec', 'ols_did', 'save_ols_did', 'single_treated_unit', 'basic_cleaning', 'multi_post_periods', 'panel_cleaned_data', 'df'}

        assert set(os.listdir(output_path)) == {'save_ols_did.txt'}

        # Ensure no nodes failed
        assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"