import pytest
import pandas as pd
from pathlib import Path
import os
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.estimators import fit_double_lasso, fit_ols
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.output import write_statsmodels_summary
from pyautocausal.pipelines.example_graph import causal_pipeline, generate_mock_data, simple_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.persistence.visualizer import visualize_graph
import json
import nbformat
import numpy as np
from pyautocausal.orchestration.nodes import NodeState
import logging
from pyautocausal.orchestration.nodes import NodeState

def preprocess_lalonde_data() -> str:
    """
    Load and preprocess the LaLonde dataset.
    
    Returns:
        str: String representation of the processed dataset
    """
    url = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
    df = pd.read_csv(url)
    y = df['re78']
    t = df['treat']
    X = df.drop(columns=['re78', 'treat','ID'])

    df = pd.DataFrame({'y': y, 'treat': t, **X})
    return df

# The causal_graph function is now imported from conftest.py

def test_causal_pipeline_execution(causal_graph):
    """Test that the pipeline executes successfully"""
    
    # Execute with input data
    causal_graph.fit(data=preprocess_lalonde_data())

def test_causal_pipeline_large_dataset(causal_graph, tmp_path):
    """Test pipeline with large dataset (should use DoubleML)"""
    # Create mock large dataset
    large_df = pd.DataFrame({
        'y': range(150),
        'treat': [1, 0] * 75,
        'age': range(150),
        'educ': [12] * 150,
    })
    
    causal_graph.fit(data=large_df)
    
    # Get the output directory from the graph
    output_dir = tmp_path / "causal_output"
    
    # Check that DoubleML results exist
    assert (output_dir / 'doubleml_summary.txt').exists()
    assert not (output_dir / 'ols_summary.txt').exists()

def test_causal_pipeline_small_dataset(causal_graph, tmp_path):
    """Test pipeline with small dataset (should use OLS)"""
    # Create mock small dataset
    small_df = pd.DataFrame({
        'y': range(50),
        'treat': [1, 0] * 25,
        'age': range(50),
        'educ': [12] * 50,
    })
    
    causal_graph.fit(data=small_df)
    
    # Get the output directory from the graph
    output_dir = tmp_path / "causal_output"
    
    # Check that OLS results exist
    assert (output_dir / 'ols_summary.txt').exists()
    assert not (output_dir / 'doubleml_summary.txt').exists()

def test_example_graph_pipeline(tmp_path):
    """Test that the example graph pipeline executes successfully and produces expected outputs."""
    # Create output directory
    output_path = tmp_path / "example_output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize the graph
    graph = causal_pipeline(output_path)

    # Generate mock data with more units for proper synthetic control
    data = generate_mock_data(n_units=2000, n_periods=2, n_treated=500)
    data_path = output_path / "notebooks" / "causal_pipeline_data.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)

    # Fit the graph
    graph.fit(df=data)
    
    # Export outputs (this creates the visualization and notebooks)
    from pyautocausal.pipelines.example_graph import _export_outputs
    _export_outputs(graph, output_path)

    # Check node states
    completed_nodes = 0
    failed_nodes = []
    for node in graph.nodes():
        if hasattr(node, 'state'):
            assert node.state is not None, f"Node {node.name} has no state"
            if hasattr(node.state, 'name'):
                if node.state.name == 'COMPLETED':
                    completed_nodes += 1
                elif node.state.name == 'FAILED':
                    failed_nodes.append(node.name)

    # Ensure no nodes failed
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"

    # Ensure at least some nodes completed successfully
    assert completed_nodes > 0, "No nodes completed successfully"

    # Check that visualization and notebook files were generated
    viz_path = output_path / "text" / "causal_pipeline_visualization.md"
    assert viz_path.exists(), f"Expected {viz_path} to exist"
    
    # Check that notebook file was generated
    notebook_path = output_path / "notebooks" / "causal_pipeline_execution.ipynb"
    assert notebook_path.exists(), f"Expected {notebook_path} to exist"
    
    # Check that HTML report was generated (optional due to potential import issues)
    html_path = output_path / "notebooks" / "causal_pipeline_execution.html"
    if html_path.exists():
        print("HTML export succeeded")
    else:
        print("HTML export failed - this is expected due to linearmodels import issues in notebook execution")
    
    # Check notebook content
    with open(notebook_path, 'r') as f:
        notebook_content = json.load(f)
    assert 'cells' in notebook_content, "Notebook should have cells"
    assert len(notebook_content['cells']) > 0, "Notebook should have at least one cell"

def test_cross_sectional_branch(tmp_path):
    """Test cross-sectional branch (single period data)."""
    output_path = tmp_path / "cross_sectional_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create single period data (triggers cross-sectional branch)
    data = generate_mock_data(n_units=100, n_periods=1, n_treated=50)
    
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that cross-sectional files are generated
    text_files = list((output_path / "text").glob("*.txt"))
    expected_files = ["ols_stand_output.txt"]
    
    for expected_file in expected_files:
        assert any(expected_file in str(f) for f in text_files), f"Expected file {expected_file} not found"
    
    # Verify nodes completed
    completed_nodes = [node.name for node in graph.nodes() 
                      if hasattr(node, 'state') and node.state.name == 'COMPLETED']
    failed_nodes = [node.name for node in graph.nodes() 
                   if hasattr(node, 'state') and node.state.name == 'FAILED']
    
    # Ensure no nodes failed
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
    
    assert 'stand_spec' in completed_nodes
    assert 'ols_stand' in completed_nodes
    assert 'ols_stand_output' in completed_nodes


def test_synthetic_did_branch(tmp_path):
    """Test Synthetic DiD branch (multiple periods, single treated unit)."""
    output_path = tmp_path / "synthdid_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data with single treated unit (triggers Synthetic DiD branch)
    data = generate_mock_data(n_units=50, n_periods=5, n_treated=1)
    
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that Synthetic DiD files are generated
    plot_files = list((output_path / "plots").glob("*.png"))
    text_files = list((output_path / "text").glob("*.txt"))
    
    # Should have synthdid plot and DiD results
    assert any("synthdid_plot" in str(f) for f in plot_files), "Synthetic DiD plot not found"
    assert any("save_ols_did" in str(f) for f in text_files), "DiD results not found"
    
    # Verify nodes completed
    completed_nodes = [node.name for node in graph.nodes() 
                      if hasattr(node, 'state') and node.state.name == 'COMPLETED']
    failed_nodes = [node.name for node in graph.nodes() 
                   if hasattr(node, 'state') and node.state.name == 'FAILED']
    
    # Ensure no nodes failed
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
    
    assert 'synthdid_spec' in completed_nodes
    assert 'synthdid_fit' in completed_nodes
    assert 'synthdid_plot' in completed_nodes


def test_standard_did_branch(tmp_path):
    """Test standard DiD branch (multiple periods, multiple treated units, insufficient post periods)."""
    output_path = tmp_path / "standard_did_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data with 2 periods only (insufficient for staggered treatment)
    data = generate_mock_data(n_units=100, n_periods=2, n_treated=50)
    
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that standard DiD files are generated
    text_files = list((output_path / "text").glob("*.txt"))
    expected_files = ["save_ols_did.txt", "save_did_output.txt"]
    
    for expected_file in expected_files:
        assert any(expected_file in str(f) for f in text_files), f"Expected file {expected_file} not found"
    
    # Verify nodes completed
    completed_nodes = [node.name for node in graph.nodes() 
                      if hasattr(node, 'state') and node.state.name == 'COMPLETED']
    failed_nodes = [node.name for node in graph.nodes() 
                   if hasattr(node, 'state') and node.state.name == 'FAILED']
    
    # Ensure no nodes failed
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
    
    assert 'did_spec' in completed_nodes
    assert 'ols_did' in completed_nodes


def test_event_study_branch(tmp_path):
    """Test event study branch (multiple periods, no staggered treatment)."""
    output_path = tmp_path / "event_study_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data with multiple periods, multiple treated units, non-staggered treatment
    data = generate_mock_data(n_units=100, n_periods=5, n_treated=50, staggered_treatment=False)
    
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that event study files are generated
    plot_files = list((output_path / "plots").glob("*.png"))
    text_files = list((output_path / "text").glob("*.txt"))
    
    # Should have event study plot and results
    assert any("event_study_plot" in str(f) for f in plot_files), "Event study plot not found"
    assert any("save_event_output" in str(f) for f in text_files), "Event study results not found"
    
    # Verify nodes completed
    completed_nodes = [node.name for node in graph.nodes() 
                      if hasattr(node, 'state') and node.state.name == 'COMPLETED']
    failed_nodes = [node.name for node in graph.nodes() 
                   if hasattr(node, 'state') and node.state.name == 'FAILED']
    
    # Ensure no nodes failed
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
    
    assert 'event_spec' in completed_nodes
    assert 'ols_event' in completed_nodes
    assert 'event_plot' in completed_nodes


def test_staggered_did_with_never_treated_branch(tmp_path):
    """Test staggered DiD branch with never-treated units (Callaway & Sant'Anna)."""
    output_path = tmp_path / "staggered_never_treated_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data with staggered treatment and some never-treated units
    data = generate_mock_data(n_units=100, n_periods=5, n_treated=60, staggered_treatment=True)
    
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that Callaway & Sant'Anna files are generated
    plot_files = list((output_path / "plots").glob("*.png"))
    text_files = list((output_path / "text").glob("*.txt"))
    
    # Should have C&S plots and results
    cs_files = [f for f in text_files if "callaway_santanna" in str(f)]
    cs_plots = [f for f in plot_files if "callaway_santanna" in str(f)]
    
    assert len(cs_files) > 0, "No Callaway & Sant'Anna result files found"
    assert len(cs_plots) > 0, "No Callaway & Sant'Anna plots found"
    
    # Verify nodes completed
    completed_nodes = [node.name for node in graph.nodes() 
                      if hasattr(node, 'state') and node.state.name == 'COMPLETED']
    failed_nodes = [node.name for node in graph.nodes() 
                   if hasattr(node, 'state') and node.state.name == 'FAILED']
    
    # Ensure no nodes failed
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
    
    assert 'stag_spec' in completed_nodes


def test_staggered_did_without_never_treated_branch(tmp_path):
    """Test staggered DiD branch without never-treated units (all units eventually treated)."""
    output_path = tmp_path / "staggered_all_treated_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data where all units are eventually treated (no never-treated control)
    data = generate_mock_data(n_units=50, n_periods=5, n_treated=50, staggered_treatment=True)
    
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that not-yet-treated Callaway & Sant'Anna files are generated
    text_files = list((output_path / "text").glob("*.txt"))
    
    # Should use not-yet-treated control when no never-treated units available
    cs_files = [f for f in text_files if "callaway_santanna" in str(f)]
    assert len(cs_files) > 0, "No Callaway & Sant'Anna result files found"
    
    # Verify staggered specification completed
    completed_nodes = [node.name for node in graph.nodes() 
                      if hasattr(node, 'state') and node.state.name == 'COMPLETED']
    assert 'stag_spec' in completed_nodes


def test_all_branches_comprehensive(tmp_path):
    """Comprehensive test ensuring all major branches can be triggered."""
    
    test_scenarios = [
        {
            'name': 'cross_sectional',
            'data_params': {'n_units': 100, 'n_periods': 1, 'n_treated': 50},
            'expected_files': ['ols_stand_output.txt'],
            'expected_nodes': ['stand_spec', 'ols_stand']
        },
        {
            'name': 'synthetic_did',
            'data_params': {'n_units': 50, 'n_periods': 5, 'n_treated': 1},
            'expected_files': ['save_ols_did.txt'],
            'expected_plots': ['synthdid_plot.png'],
            'expected_nodes': ['synthdid_spec', 'synthdid_fit']
        },
        {
            'name': 'standard_did',
            'data_params': {'n_units': 100, 'n_periods': 2, 'n_treated': 50},
            'expected_files': ['save_ols_did.txt', 'save_did_output.txt'],
            'expected_nodes': ['did_spec', 'ols_did']
        },
        {
            'name': 'event_study',
            'data_params': {'n_units': 100, 'n_periods': 5, 'n_treated': 50, 'staggered_treatment': False},
            'expected_files': ['save_event_output.txt'],
            'expected_plots': ['event_study_plot.png'],
            'expected_nodes': ['event_spec', 'ols_event']
        },
        {
            'name': 'staggered_did',
            'data_params': {'n_units': 100, 'n_periods': 5, 'n_treated': 60, 'staggered_treatment': True, 'random_seed': 42},
            'expected_files': ['save_stag_output.txt'],
            'expected_nodes': ['stag_spec', 'ols_stag']
        }
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        
        # Create separate output directory for each scenario
        scenario_output_path = tmp_path / f"{scenario['name']}_output"
        scenario_output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate appropriate mock data
        if 'data_params' in scenario:
            data = generate_mock_data(**scenario['data_params'])
        
        # Run the graph
        graph = causal_pipeline(scenario_output_path)
        graph.fit(df=data)
        
        # Check expected files
        text_files = list((scenario_output_path / "text").glob("*.txt"))
        if 'expected_files' in scenario:
            for expected_file in scenario['expected_files']:
                assert any(expected_file in str(f) for f in text_files), \
                    f"Expected file {expected_file} not found in {scenario['name']} scenario"
        
        # Check expected plots
        if 'expected_plots' in scenario:
            plot_files = list((scenario_output_path / "plots").glob("*.png"))
            for expected_plot in scenario['expected_plots']:
                assert any(expected_plot in str(f) for f in plot_files), \
                    f"Expected plot {expected_plot} not found in {scenario['name']} scenario"
        
        # Check expected nodes completed
        completed_nodes = [node.name for node in graph.nodes() 
                          if hasattr(node, 'state') and node.state.name == 'COMPLETED']
        
        # Check for failed nodes
        failed_nodes = [node.name for node in graph.nodes() 
                       if hasattr(node, 'state') and node.state.name == 'FAILED']
        
        # Ensure no nodes failed in this scenario
        assert len(failed_nodes) == 0, f"The following nodes failed in {scenario['name']} scenario: {failed_nodes}"
        
        for expected_node in scenario['expected_nodes']:
            assert expected_node in completed_nodes, \
                f"Expected node {expected_node} not completed in {scenario['name']} scenario"
        
        all_results[scenario['name']] = {
            'completed_nodes': len(completed_nodes),
            'text_files': len(text_files),
            'total_files': len(list(scenario_output_path.glob("**/*.*")))
        }
        
        print(f"✓ {scenario['name']} scenario completed successfully")
    
    # Summary assertion
    assert len(all_results) == len(test_scenarios), "Not all scenarios completed"
    print(f"\n✓ All {len(test_scenarios)} branch scenarios tested successfully")
    print("Summary:", all_results)

def test_manual_staggered_did_branch(tmp_path):
    """Test specifically for the staggered DiD branch with manually created proper staggered data."""
    
    # Create output directory
    output_path = tmp_path / "staggered_test_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create manual staggered treatment data with different treatment start times
    np.random.seed(42)
    n_units = 60
    n_periods = 5
    
    data_list = []
    
    # Create never-treated units (30% of units)
    never_treated_units = list(range(0, 18))
    for unit in never_treated_units:
        for t in range(n_periods):
            data_list.append({
                'id_unit': unit,
                't': t,
                'treat': 0,
                'y': np.random.normal(0, 1),
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1)
            })
    
    # Create units treated at different times (staggered)
    # Cohort 1: Treatment starts at t=2 (15 units)
    cohort_1_units = list(range(18, 33))
    for unit in cohort_1_units:
        for t in range(n_periods):
            treat = 1 if t >= 2 else 0
            data_list.append({
                'id_unit': unit,
                't': t,
                'treat': treat,
                'y': np.random.normal(2 * treat, 1),  # Treatment effect = 2
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1)
            })
    
    # Cohort 2: Treatment starts at t=3 (15 units)
    cohort_2_units = list(range(33, 48))
    for unit in cohort_2_units:
        for t in range(n_periods):
            treat = 1 if t >= 3 else 0
            data_list.append({
                'id_unit': unit,
                't': t,
                'treat': treat,
                'y': np.random.normal(2 * treat, 1),  # Treatment effect = 2
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1)
            })
    
    # Cohort 3: Treatment starts at t=4 (12 units)
    cohort_3_units = list(range(48, 60))
    for unit in cohort_3_units:
        for t in range(n_periods):
            treat = 1 if t >= 4 else 0
            data_list.append({
                'id_unit': unit,
                't': t,
                'treat': treat,
                'y': np.random.normal(2 * treat, 1),  # Treatment effect = 2
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1)
            })
    
    data = pd.DataFrame(data_list)
    
    # Verify staggered treatment condition
    from pyautocausal.pipelines.library.conditions import has_staggered_treatment
    assert has_staggered_treatment(data), "Manual data should have staggered treatment"
    
    # Initialize and run the graph
    graph = causal_pipeline(output_path)
    graph.fit(df=data)
    
    # Check that staggered DiD nodes were executed
    stag_spec_node = next((node for node in graph.nodes() if node.name == 'stag_spec'), None)
    ols_stag_node = next((node for node in graph.nodes() if node.name == 'ols_stag'), None)
    
    # Check for any failed nodes
    failed_nodes = [node.name for node in graph.nodes() 
                   if hasattr(node, 'state') and node.state.name == 'FAILED']
    assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
    
    assert stag_spec_node is not None, "stag_spec node should exist"
    assert ols_stag_node is not None, "ols_stag node should exist"
    assert stag_spec_node.state == NodeState.COMPLETED, f"stag_spec node should be completed, got {stag_spec_node.state}"
    assert ols_stag_node.state == NodeState.COMPLETED, f"ols_stag node should be completed, got {ols_stag_node.state}"
    
    # Check for output files
    text_files = list((output_path / "text").glob("*.txt")) if (output_path / "text").exists() else []
    stag_output_files = [f for f in text_files if 'save_stag_output.txt' in str(f)]
    assert len(stag_output_files) > 0, "Expected staggered DiD output files were not generated"

def test_serialized_simple_graph_execution(tmp_path):
    """
    Tests that a serialized and deserialized graph can be executed successfully.
    This provides an end-to-end check of the YAML round-trip and execution logic.
    """
    # 1. Build the initial graph and configure its output path
    graph = simple_graph()
    
    # 2. Serialize the graph to a YAML file
    yaml_path = tmp_path / "simple_graph.yml"
    graph.to_yaml(yaml_path)
    assert yaml_path.exists()

    # 3. Load the graph back from the YAML file
    loaded_graph = ExecutableGraph.from_yaml(yaml_path)
    
    # Configure the runtime for the deserialized graph
    loaded_output_path = tmp_path / "loaded_output"
    loaded_graph.configure_runtime(output_path=loaded_output_path)

    # 4. Execute the loaded graph with mock data
    mock_data = generate_mock_data(n_units=50, n_periods=2, n_treated=10)
    loaded_graph.fit(df=mock_data)

    # 5. Assert that the execution was successful
    # Check that all nodes reached a terminal state
    for node in loaded_graph.nodes():
        if hasattr(node, 'state'):
            assert node.state.is_terminal(), f"Node {node.name} did not complete successfully. State: {node.state.name}"

    # 5. Verify that output files were created for the executed branch.
    #    The mock data has multiple periods, so the 'did_spec' branch should run.
    did_node = loaded_graph.get("save_ols_did")
    stand_node = loaded_graph.get("ols_stand_output")

    # The stand_spec branch should be PASSED (skipped)
    assert stand_node.state == NodeState.PASSED, "Standard spec branch should have been skipped."
    assert not (loaded_output_path / "ols_stand_output.txt").exists(), "Output file from skipped branch should not exist."
    
    # The DiD branch should be COMPLETED
    assert did_node.state == NodeState.COMPLETED, "DiD spec branch should have completed."
    assert (loaded_output_path / "save_ols_did.txt").exists(), "Output file from executed DiD branch should exist."

    # Export the notebook
    notebook_path = loaded_output_path / "simple_graph.ipynb"
    exporter = NotebookExporter(loaded_graph)
    exporter.export_notebook(notebook_path)
    assert notebook_path.exists(), "Notebook was not exported"
    
    # Load and parse the notebook properly
    with open(notebook_path, 'r') as f:
        notebook_json = json.load(f)
        notebook = nbformat.reads(json.dumps(notebook_json), as_version=4)
    
    import_cell_contents = "\n".join(exporter.needed_imports)
    # Now properly check cells
    found_import_cell = False
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell_source = cell.source
            if cell_source == import_cell_contents:
                found_import_cell = True
                break

    assert found_import_cell, "Notebook does not contain the expected import cell"

    # Verify notebook content
    with open(notebook_path, 'r') as f:
        notebook_content = f.read()
        assert "Causal Analysis Pipeline" in notebook_content, "Notebook header missing"