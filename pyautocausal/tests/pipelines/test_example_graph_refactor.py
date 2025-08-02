"""Tests for the refactored example_graph structure.

This test module verifies that:
1. The new modular structure imports correctly
2. Graph creation functions work
3. Backward compatibility is maintained
4. The core decision logic is sound
"""

import pytest
import tempfile
from pathlib import Path

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.example_graph import (
    create_panel_graph, 
    create_cross_sectional_graph,
    main
)
from pyautocausal.pipelines.example_graph.core import (
    create_panel_decision_structure,
    configure_panel_decision_paths,
    create_cross_sectional_decision_structure,
    configure_cross_sectional_decision_paths
)
from pyautocausal.pipelines.example_graph.branches import (
    add_did_and_event_study_branches,
    add_callaway_and_santanna_branches,
    add_synthdid_branch,
    add_cross_sectional_branches
)


class TestGraphCreationRefactor:
    """Test suite for the refactored graph creation logic."""

    def test_imports(self):
        """Test that all refactored components can be imported."""
        assert callable(create_panel_graph)
        assert callable(create_cross_sectional_graph)
        assert callable(main)
        assert callable(create_panel_decision_structure)
        assert callable(add_did_and_event_study_branches)

    def test_panel_graph_creation(self):
        """Test that the panel graph is created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            graph = create_panel_graph(output_path, output_path)
            
            assert isinstance(graph, ExecutableGraph)
            node_names = {node.name for node in graph.nodes()}
            
            # Check for essential panel nodes
            assert 'panel_cleaned_data' in node_names
            assert 'synthdid_spec' in node_names
            assert 'did_spec' in node_names
            
            # Ensure no cross-sectional nodes are present
            assert 'cross_sectional_cleaned_data' not in node_names
            assert 'ate_spec' not in node_names

    def test_cross_sectional_graph_creation(self):
        """Test that the cross-sectional graph is created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            graph = create_cross_sectional_graph(output_path, output_path)
            
            assert isinstance(graph, ExecutableGraph)
            node_names = {node.name for node in graph.nodes()}
            
            # Check for essential cross-sectional nodes
            assert 'cross_sectional_cleaned_data' in node_names
            assert 'ate_spec' in node_names
            assert 'cate_spec' in node_names
            
            # Ensure no panel nodes are present
            assert 'panel_cleaned_data' not in node_names
            assert 'synthdid_spec' not in node_names

    def test_node_separation(self):
        """Test that panel and cross-sectional graphs are mutually exclusive in their nodes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            panel_graph = create_panel_graph(output_path, output_path)
            cs_graph = create_cross_sectional_graph(output_path, output_path)
            
            panel_nodes = {node.name for node in panel_graph.nodes()}
            cs_nodes = {node.name for node in cs_graph.nodes()}
            
            # Find common nodes (should just be the shared head)
            common_nodes = panel_nodes.intersection(cs_nodes)
            assert common_nodes == {'df', 'basic_cleaning'}
            
            # Check for cross-contamination
            assert 'ate_spec' not in panel_nodes
            assert 'did_spec' not in cs_nodes 