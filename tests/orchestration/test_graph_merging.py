import pytest
import pandas as pd
from pyautocausal.orchestration.graph_builder import GraphBuilder
from pyautocausal.orchestration.nodes import Node, InputNode
import copy
from inspect import Parameter, Signature

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def test_merge_linked_graphs():
    """Test merging graphs that are properly linked"""
    graph1Builder = GraphBuilder()
    graph2Builder = GraphBuilder()

    # Create and link nodes
    graph1Builder.create_node(name="process1", action_function=process_dataframe)
    graph2Builder.add_input_node(name="input2", input_dtype=pd.DataFrame)

    graph1 = graph1Builder.build()
    graph2 = graph2Builder.build()
    
    # Merge graphs with explicit wiring
    merged = graph1.merge_with(graph2, graph1.get("process1") >> graph2.get("input2"))
    
    # Verify structure
    new_input = next(n for n in merged.nodes() if n.name == "input2")
    assert new_input in merged.nodes()
    assert graph1.get("process1") in merged.nodes()
    assert new_input in graph1.get_node_successors(graph1.get("process1"))
    assert new_input.graph == merged

def test_fit_with_merged_graphs():
    """Test fit behavior with merged graphs"""
    graph1 = (GraphBuilder()
    .add_input_node(name="external_input", input_dtype=pd.DataFrame)
    .create_node(name="process", action_function=process_dataframe, predecessors={"df": "external_input"})
    .build())
    
    graph2 = (GraphBuilder()
    .add_input_node(name="internal_input", input_dtype=pd.DataFrame)
    .create_node(name="transform", action_function=transform_dataframe, predecessors={"df": "internal_input"})
    .build())
    
    # Merge graphs with explicit wiring
    graph1.merge_with(graph2, graph1.get("process") >> graph2.get("internal_input"))

    assert len(graph1.input_nodes) == 1
    
    # Should only need to provide external input
    test_df = pd.DataFrame({'a': [1, 2, 3]})
    

    graph1.fit(external_input=test_df)
    
    # Verify execution
    new_input = graph1.get("internal_input")
    new_transform = graph1.get("transform")
    assert new_input.is_completed()
    assert new_transform.is_completed()

def test_fit_with_multiple_external_inputs():
    """Builed two graphs. Graph 1 has one integer input, squares it and then has one output.
    and then has one output. Graph 2 takes two integer inputs and adds them together.
    Wire graph 2 to graph 1 and verify that the merge graph has two inputs and one output."""
    def square(x: int) -> int:
        return x**2
    
    def add(x: int, y: int) -> int:
        return x + y
    
    graph1 = (GraphBuilder()
    .add_input_node(name="input1", input_dtype=int)
    .create_node(name="square", action_function=square, predecessors={"x": "input1"})
    .build())
    
    graph2 = (GraphBuilder()
    .add_input_node(name="input2", input_dtype=int)
    .add_input_node(name="input3", input_dtype=int)
    .create_node(name="add", action_function=add, predecessors={"x": "input2", "y": "input3"})
    .build())
    
    graph1.merge_with(graph2, graph1.get("square") >> graph2.get("input2"))

    graph1.fit(input1=2, input3=3)
    assert graph1.get("add").is_completed()
    assert graph1.get("add").output == 7
    
    assert len(graph1.input_nodes) == 2

def test_merge_with_non_pending_nodes():
    def square(x: int) -> int:
        return x**2
    
    def add(x: int, y: int) -> int:
        return x + y
    
    graph1 = (GraphBuilder()
    .add_input_node(name="input1", input_dtype=int)
    .create_node(name="square", action_function=square, predecessors={"x": "input1"})
    .build())
    
    graph2 = (GraphBuilder()
    .add_input_node(name="input2", input_dtype=int)
    .add_input_node(name="input3", input_dtype=int)
    .create_node(name="add", action_function=add, predecessors={"x": "input2", "y": "input3"})
    .build())

    graph2.fit(input2=2, input3=3)
    
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, graph1.get("square") >> graph2.get("input2"))
    assert "not in PENDING state" in str(exc_info.value)
    

def test_merge_with_duplicate_node_names():
    """Test that merging graphs with duplicate node names creates
    new nodes with unique names"""
    def add_one(x: int) -> int:
        return x + 1
    
    graph1 = (GraphBuilder()
    .add_input_node(name="input1", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input1"})
    .build())

    graph2 = (GraphBuilder()
    .add_input_node(name="input2", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input2"})
    .build())

    graph1.merge_with(graph2, graph1.get("add_one") >> graph2.get("input2"))

    assert len(graph1.nodes()) == 4
    assert "add_one" in graph1._nodes_by_name
    assert "add_one_1" in graph1._nodes_by_name

    graph1.fit(input1=1)
    assert graph1.get("add_one_1").is_completed()
    assert graph1.get("add_one_1").output == 3


def test_merge_with_non_input_target():
    """Test that merging fails when target node is not an InputNode"""
    def add_one(x: int) -> int:
        return x + 1
    
    graph1 = (GraphBuilder()
    .add_input_node(name="input1", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input1"})
    .build())

    graph2 = (GraphBuilder()
    .add_input_node(name="input2", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input2"})
    .build())

    
    # Attempt to wire regular nodes should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, graph1.get("add_one") >> graph2.get("add_one"))
    assert "Target node must be an input node" in str(exc_info.value)

def test_merge_with_wrong_graph_nodes():
    """Test that merging fails when nodes are from wrong graphs"""
    def add_one(x: int) -> int:
        return x + 1
    
    graph1 = (GraphBuilder()
    .add_input_node(name="input1", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input1"})
    .build())

    graph2 = (GraphBuilder()
    .add_input_node(name="input2", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input2"})
    .build())

    graph3 = (GraphBuilder()
    .add_input_node(name="input3", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors={"x": "input3"})
    .build())

    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, graph1.get("add_one") >> graph3.get("input3"))
    assert str(exc_info.value).startswith("Invalid wiring")



def test_merge_preserves_right_hand_graph():
    """This test does nothing because I can't figure out how to copy a graph
    TODO: Test this
    """
    pass
    
def test_merge_with_no_wirings():
    """Test that merging without any wirings fails"""
    graph1 = GraphBuilder().build()
    graph2 = GraphBuilder().build()
    
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2)
    assert "At least one wiring" in str(exc_info.value)