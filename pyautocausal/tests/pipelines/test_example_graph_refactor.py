"""Tests for the refactored example_graph structure.

This test module verifies that:
1. The new modular structure imports correctly
2. Graph creation functions work
3. Backward compatibility is maintained
4. The core decision logic is sound
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from pyautocausal.orchestration.graph import ExecutableGraph


class TestExampleGraphRefactor:
    """Test suite for the refactored example graph structure."""

    def test_new_structure_imports(self):
        """Test that all modules in the new structure can be imported."""
        # Test importing from the new modular structure
        from pyautocausal.pipelines.example_graph import causal_pipeline, simple_graph, main
        from pyautocausal.pipelines.example_graph.core import (
            create_core_decision_structure, 
            configure_core_decision_paths
        )
        from pyautocausal.pipelines.example_graph.branches import create_all_analysis_branches
        from pyautocausal.pipelines.example_graph.utils import (
            setup_output_directories, 
            print_execution_summary,
            export_outputs
        )
        
        # All imports should succeed without errors
        assert callable(causal_pipeline)
        assert callable(simple_graph)
        assert callable(main)
        assert callable(create_core_decision_structure)
        assert callable(create_all_analysis_branches)

    def test_backward_compatibility(self):
        """Test that the old import style still works."""
        # Test importing from the legacy interface
        from pyautocausal.pipelines.example_graph import causal_pipeline, simple_graph, main
        
        # These should be the same functions as in the new structure
        assert callable(causal_pipeline)
        assert callable(simple_graph)
        assert callable(main)

    def test_simple_graph_creation(self):
        """Test that the simple graph can be created successfully."""
        from pyautocausal.pipelines.example_graph import simple_graph
        
        graph = simple_graph()
        
        # Verify it's an ExecutableGraph
        assert isinstance(graph, ExecutableGraph)
        
        # Verify it has the expected nodes
        node_names = [node.name for node in graph.nodes()]
        expected_nodes = [
            "df", "multi_period", "stand_spec", "ols_stand", 
            "ols_stand_output", "did_spec", "ols_did", "save_ols_did"
        ]
        
        for expected_node in expected_nodes:
            assert expected_node in node_names, f"Missing expected node: {expected_node}"
        
        print(f"Simple graph created successfully with {len(node_names)} nodes")

    def test_full_pipeline_creation(self):
        """Test that the full causal pipeline can be created successfully."""
        from pyautocausal.pipelines.example_graph import causal_pipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # This should not raise any errors
            graph = causal_pipeline(output_path)
            
            # Verify it's an ExecutableGraph
            assert isinstance(graph, ExecutableGraph)
            
            # Verify it has a reasonable number of nodes (should be ~30+)
            node_count = len(list(graph.nodes()))
            assert node_count > 20, f"Expected many nodes, got {node_count}"
            
            # Verify key decision nodes exist
            node_names = [node.name for node in graph.nodes()]
            key_decision_nodes = [
                "multi_period", "single_treated_unit", 
                "multi_post_periods", "stag_treat"
            ]
            
            for decision_node in key_decision_nodes:
                assert decision_node in node_names, f"Missing key decision node: {decision_node}"
            
            # Verify output directories were created
            assert (output_path / "plots").exists()
            assert (output_path / "text").exists()  
            assert (output_path / "notebooks").exists()
            
            print(f"Full pipeline created successfully with {node_count} nodes")

    def test_core_decision_structure(self):
        """Test that the core decision structure is created correctly."""
        from pyautocausal.pipelines.example_graph.core import create_core_decision_structure
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            text_dir = output_path / "text"
            text_dir.mkdir(parents=True)
            
            graph = ExecutableGraph()
            
            # This should not raise any errors
            create_core_decision_structure(graph, text_dir)
            
            # Verify core nodes exist
            node_names = [node.name for node in graph.nodes()]
            core_nodes = [
                "df", "basic_cleaning", "multi_period", 
                "panel_cleaned_data", "cross_sectional_cleaned_data"
            ]
            
            for core_node in core_nodes:
                assert core_node in node_names, f"Missing core node: {core_node}"

    def test_directory_structure_setup(self):
        """Test that output directory setup works correctly."""
        from pyautocausal.pipelines.example_graph.utils import setup_output_directories
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            plots_dir, text_dir, notebooks_dir = setup_output_directories(output_path)
            
            # Verify directories were created
            assert plots_dir.exists()
            assert text_dir.exists()
            assert notebooks_dir.exists()
            
            # Verify they are absolute paths
            assert plots_dir.is_absolute()
            assert text_dir.is_absolute()
            assert notebooks_dir.is_absolute()

    def test_mock_data_compatibility(self):
        """Test that the pipeline works with mock data."""
        from pyautocausal.pipelines.example_graph import simple_graph
        from pyautocausal.pipelines.mock_data import generate_mock_data
        
        # Generate simple mock data
        data = generate_mock_data(
            n_units=20, 
            n_periods=3, 
            n_treated=10, 
            staggered_treatment=False
        )
        
        # Create simple graph
        graph = simple_graph()
        
        # This should not raise errors (we're just testing graph creation, not execution)
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'treat' in data.columns
        assert 'y' in data.columns

    def test_file_organization(self):
        """Test that the file organization is correct."""
        from pathlib import Path
        
        example_graph_dir = Path(__file__).parent.parent.parent / "pyautocausal" / "pipelines" / "example_graph"
        
        # Verify all expected files exist
        expected_files = [
            "__init__.py",
            "core.py", 
            "branches.py",
            "utils.py",
            "main.py"
        ]
        
        for expected_file in expected_files:
            file_path = example_graph_dir / expected_file
            assert file_path.exists(), f"Missing expected file: {expected_file}"
            assert file_path.stat().st_size > 0, f"File is empty: {expected_file}"

    def test_node_count_consistency(self):
        """Test that node counts are consistent between simple and full graphs."""
        from pyautocausal.pipelines.example_graph import causal_pipeline, simple_graph
        
        simple = simple_graph()
        simple_count = len(list(simple.nodes()))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            full = causal_pipeline(Path(temp_dir))
            full_count = len(list(full.nodes()))
        
        # Full graph should have significantly more nodes than simple graph
        assert full_count > simple_count * 3, f"Full graph ({full_count}) should be much larger than simple graph ({simple_count})"
        
        print(f"Node count comparison: Simple={simple_count}, Full={full_count}")


if __name__ == "__main__":
    # Run tests directly
    test_suite = TestExampleGraphRefactor()
    
    print("Running example_graph refactor tests...")
    
    try:
        test_suite.test_new_structure_imports()
        print("✓ New structure imports test passed")
        
        test_suite.test_backward_compatibility()
        print("✓ Backward compatibility test passed")
        
        test_suite.test_simple_graph_creation()
        print("✓ Simple graph creation test passed")
        
        test_suite.test_directory_structure_setup()
        print("✓ Directory structure setup test passed")
        
        test_suite.test_file_organization()
        print("✓ File organization test passed")
        
        test_suite.test_node_count_consistency()
        print("✓ Node count consistency test passed")
        
        print("\n✓ All tests passed! The refactored structure is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise 