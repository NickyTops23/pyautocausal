from pathlib import Path

import pytest

from pyautocausal.pipelines.example_graph import simple_graph
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.persistence.notebook_export import NotebookExporter
import nbformat


def _build_graph(tmp_dir: Path):
    """Helper to create the example graph inside *tmp_dir*."""
    return simple_graph(tmp_dir / "output")


def test_yaml_roundtrip(tmp_path):
    """Serialize a graph to YAML and load it back, validating integrity."""
    graph = _build_graph(tmp_path)

    yaml_path = tmp_path / "graph.yml"
    graph.to_yaml(yaml_path)

    loaded = ExecutableGraph.from_yaml(yaml_path)

    # Runtime configuration is not serialized; configure for the tmp dir
    loaded.configure_runtime(output_path=tmp_path / "loaded_output")

    # Structural checks
    assert len(graph.nodes()) == len(loaded.nodes())
    assert {n.name for n in graph.nodes()} == {n.name for n in loaded.nodes()}
    assert {
        (u.name, v.name) for u, v in graph.edges()
    } == {(u.name, v.name) for u, v in loaded.edges()}

    # Ensure the loaded graph can execute successfully
    df = generate_mock_data(n_units=100, n_periods=2, n_treated=20)
    loaded.fit(df=df)

    # All nodes should have completed or passed after execution
    for node in loaded.nodes():
        assert node.state.is_terminal()  # type: ignore[attr-defined] 


def test_yaml_roundtrip_notebook_export(tmp_path):
    """After YAML round-trip, ensure we can export a notebook successfully."""
    graph = _build_graph(tmp_path)

    # Save + load via YAML
    yaml_path = tmp_path / "graph.yml"
    graph.to_yaml(yaml_path)
    loaded = ExecutableGraph.from_yaml(yaml_path)
    loaded.configure_runtime(output_path=tmp_path / "loaded_output")

    # Execute so that nodes are completed (needed for exporter)
    df = generate_mock_data(n_units=100, n_periods=2, n_treated=20)
    loaded.fit(df=df)

    # Export notebook
    nb_path = tmp_path / "roundtrip.ipynb"
    exporter = NotebookExporter(loaded)
    exporter.export_notebook(str(nb_path))

    # Basic sanity check – notebook file exists and is parseable
    assert nb_path.exists()
    with nb_path.open() as fh:
        nb = nbformat.read(fh, as_version=4)
    # header markdown cell expected
    assert nb.cells and nb.cells[0].cell_type == "markdown" 