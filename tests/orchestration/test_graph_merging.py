import pytest
import pandas as pd
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.orchestration.nodes import Node, InputNode

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def test_merge_linked_graphs():
    """Test merging graphs that are properly linked"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create and link nodes
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph1)
    input_node2 = InputNode(name="input2", graph=graph2, input_dtype=pd.DataFrame)
    
    # Merge graphs with explicit wiring
    merged = graph1.merge_with(graph2, node1 >> input_node2)
    
    # Verify structure
    new_input = next(n for n in merged.nodes() if n.name == "input2")
    assert new_input in merged.nodes()
    assert node1 in merged.nodes()
    assert new_input in node1.get_successors()
    assert new_input.graph == merged

def test_fit_with_merged_graphs():
    """Test fit behavior with merged graphs"""
    graph1 = (GraphBuilder()
    .add_input_node(name="external_input", input_dtype=pd.DataFrame)
    .add_node(name="process", action_function=process_dataframe, predecessors={"df": "external_input"})
    .build())
    
    graph2 = (GraphBuilder()
    .add_input_node(name="internal_input", input_dtype=pd.DataFrame)
    .add_node(name="transform", action_function=transform_dataframe, predecessors={"df": "internal_input"})
    .build())
    
    # Merge graphs with explicit wiring
    graph1.merge_with(graph2, process_node >> input_node2)

    assert len(graph1.input_nodes) == 2
    
    # Should only need to provide external input
    test_df = pd.DataFrame({'a': [1, 2, 3]})
    

    graph1.fit(external_input=test_df)
    
    # Verify execution
    assert input_node1.is_completed()
    assert process_node.is_completed()
    new_input = next(n for n in graph1.nodes() if n.name == "internal_input")
    new_transform = next(n for n in graph1.nodes() if n.name == "transform")
    assert new_input.is_completed()
    assert new_transform.is_completed()

def test_fit_with_multiple_external_inputs():
    """Test fit with multiple external inputs in merged graphs"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create two parallel pipelines that merge
    input1 = InputNode(name="input1", graph=graph1, input_dtype=pd.DataFrame)
    input2 = InputNode(name="input2", graph=graph1, input_dtype=pd.DataFrame)
    process = Node(name="process", action_function=process_dataframe, graph=graph1)
    internal_input = InputNode(name="internal", graph=graph2, input_dtype=pd.DataFrame)
    
    # Wire nodes within their own graphs
    graph1.add_edge(input1, process)
    graph1.add_edge(input2, process)
    
    # Merge graphs with explicit wiring
    merged = graph1.merge_with(graph2, process >> internal_input)
    
    # Should need both external inputs
    test_df1 = pd.DataFrame({'a': [1, 2, 3]})
    test_df2 = pd.DataFrame({'b': [4, 5, 6]})
    merged.fit(input1=test_df1, input2=test_df2)
    
    # Verify execution
    assert input1.is_completed()
    assert input2.is_completed()
    assert process.is_completed()
    new_internal = next(n for n in merged.nodes() if n.name == "internal")
    assert new_internal.is_completed()

def test_merge_with_non_pending_nodes():
    """Test that merging fails when target graph has non-pending nodes"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create and execute a node in graph1
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph1)
    graph1.execute_graph()  # This will change node1's state to COMPLETED
    
    # Create and link nodes for merge
    input_node2 = InputNode(name="input2", graph=graph2, input_dtype=pd.DataFrame)
    node1 >> input_node2
    
    # Attempt to merge should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2)
    assert "not in PENDING state" in str(exc_info.value)

def test_merge_with_duplicate_node_names():
    """Test that merging graphs with duplicate node names raises an error"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create nodes with same name in both graphs
    node1 = Node(name="process", action_function=process_dataframe, graph=graph1)
    node2 = Node(name="process", action_function=process_dataframe, graph=graph2)
    
    # Create link between graphs
    input_node = InputNode(name="input", graph=graph2, input_dtype=pd.DataFrame)
    node1 >> input_node
    
    # Attempt to merge should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2)
    assert "duplicate node names" in str(exc_info.value)
    
    input_node2 = InputNode(name="input2", graph=graph2, input_dtype=pd.DataFrame)
    node1 >> input_node2
    
    # Merge graphs
    merged = graph1.merge_with(graph2)
    
    # Verify node configuration is preserved
    merged_node = next(n for n in merged.nodes() if n.name == "process1")
    assert merged_node.condition is not None
    assert merged_node.condition.description == "Always true"
    assert merged_node.output_config.output_type == OutputType.PARQUET

def test_merge_with_multiple_wirings():
    """Test merging with multiple explicit wirings"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create nodes in first graph
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph1)
    node2 = Node(name="process2", action_function=process_dataframe, graph=graph1)
    
    # Create nodes in second graph
    input1 = InputNode(name="input1", graph=graph2, input_dtype=pd.DataFrame)
    input2 = InputNode(name="input2", graph=graph2, input_dtype=pd.DataFrame)
    
    # Merge with multiple wirings
    merged = graph1.merge_with(graph2,
        node1 >> input1,
        node2 >> input2
    )
    
    # Verify all wirings are correct
    new_input1 = next(n for n in merged.nodes() if n.name == "input1")
    new_input2 = next(n for n in merged.nodes() if n.name == "input2")
    assert new_input1 in node1.get_successors()
    assert new_input2 in node2.get_successors()

def test_merge_with_non_input_target():
    """Test that merging fails when target node is not an InputNode"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create regular nodes in both graphs
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph1)
    node2 = Node(name="process2", action_function=process_dataframe, graph=graph2)
    
    # Attempt to wire regular nodes should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, node1 >> node2)
    assert "Target must be an InputNode" in str(exc_info.value)

def test_merge_with_wrong_graph_nodes():
    """Test that merging fails when nodes are from wrong graphs"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    graph3 = GraphBuilder().build()
    
    # Create nodes in different graphs
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph3)
    input2 = InputNode(name="input2", graph=graph2, input_dtype=pd.DataFrame)
    
    # Attempt to merge with node from wrong graph should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, node1 >> input2)
    assert "Source must be from the target graph" in str(exc_info.value)

def test_merge_preserves_original_graphs():
    """Test that merging doesn't modify the original graphs"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create and wire nodes
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph1)
    input2 = InputNode(name="input2", graph=graph2, input_dtype=pd.DataFrame)
    
    # Store original graph states
    g1_original_nodes = set(graph1.nodes())
    g2_original_nodes = set(graph2.nodes())
    
    # Merge graphs
    merged = graph1.merge_with(graph2, node1 >> input2)
    
    # Verify original graphs are unchanged
    assert set(graph1.nodes()) == g1_original_nodes
    assert set(graph2.nodes()) == g2_original_nodes
    assert input2.graph == graph2  # Original input node still belongs to graph2

def test_merge_with_no_wirings():
    """Test that merging without any wirings fails"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2)
    assert "At least one wiring" in str(exc_info.value)

def test_merge_with_reversed_wiring():
    """Test that merging fails when wiring direction is reversed"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create nodes
    node1 = Node(name="process1", action_function=process_dataframe, graph=graph2)
    input2 = InputNode(name="input2", graph=graph1, input_dtype=pd.DataFrame)
    
    # Attempt to merge with reversed wiring should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, node1 >> input2)
    assert "Invalid wiring direction" in str(exc_info.value)

def test_merge_with_name_conflicts():
    """Test that nodes are properly renamed when there are name conflicts"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    # Create nodes with same names in both graphs
    node1 = Node(name="process", action_function=process_dataframe, graph=graph1)
    node2 = Node(name="process", action_function=process_dataframe, graph=graph2)
    input2 = InputNode(name="input", graph=graph2, input_dtype=pd.DataFrame)
    
    # Wire nodes within graph2
    graph2.add_edge(input2, node2)
    
    # Merge graphs
    merged = graph1.merge_with(graph2, node1 >> input2)
    
    # Verify node was renamed
    assert node1.name == "process"  # Original node name unchanged
    renamed_input = next(n for n in merged.nodes() if isinstance(n, InputNode))
    renamed_process = next(
        n for n in merged.nodes() 
        if isinstance(n, Node) and n.name != "process"
    )
    assert renamed_input.name == "input_1"  # Input node was renamed
    assert renamed_process.name == "process_1"  # Process node was renamed
    
    # Verify edges are preserved with renamed nodes
    assert renamed_process in merged.successors(renamed_input) 