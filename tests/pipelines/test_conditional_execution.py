import pytest
from pyautocausal.orchestration.nodes import Node, NodeState
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.condition import Condition

def always_true() -> bool:
    return True

def always_false() -> bool:
    return False

def true_branch() -> str:
    return "true_branch_executed"

def false_branch() -> str:
    return "false_branch_executed"

def is_true(x: bool) -> bool:
    return x

def is_false(x: bool) -> bool:
    return not x

def final_node_action() -> str:
    return "final_node_executed"

# Create reusable conditions
true_condition = Condition(is_true, "Condition is true")
false_condition = Condition(is_false, "Condition is false")

def test_true_condition():
    """Test that when condition is True, only true branch executes"""
    graph = ExecutableGraph()
    
    # Create nodes
    condition_node = Node(
        name="condition",
        action_function=always_true
    )
    graph.add_node_to_graph(condition_node)
    
    true_node = Node(
        name="true_branch",
        action_function=true_branch,
        condition=true_condition
    )
    graph.add_node_to_graph(true_node)
    graph.add_edge(condition_node, true_node, argument_name="x")
    
    false_node = Node(
        name="false_branch",
        action_function=false_branch,
        condition=false_condition
    )
    graph.add_node_to_graph(false_node)
    graph.add_edge(condition_node, false_node, argument_name="x")
    
    graph.execute_graph()
    
    assert condition_node.is_completed()
    assert condition_node.output is True
    assert true_node.is_completed()
    assert true_node.output == "true_branch_executed"
    assert false_node.is_skipped()
    assert false_node.output is None

def test_false_condition():
    """Test that when condition is False, only false branch executes"""
    graph = ExecutableGraph()
    
    condition_node = Node(
        name="condition",
        action_function=always_false
    )
    graph.add_node_to_graph(condition_node)
    
    true_node = Node(
        name="true_branch",
        action_function=true_branch,
        condition=true_condition
    )
    graph.add_node_to_graph(true_node)
    graph.add_edge(condition_node, true_node, argument_name="x")
    
    false_node = Node(
        name="false_branch",
        action_function=false_branch,
        condition=false_condition
    )
    graph.add_node_to_graph(false_node)
    graph.add_edge(condition_node, false_node, argument_name="x")
    
    graph.execute_graph()
    
    assert condition_node.is_completed()
    assert condition_node.output is False
    assert true_node.is_skipped()
    assert true_node.is_skipped()
    assert true_node.output is None
    assert false_node.is_completed()
    assert false_node.output == "false_branch_executed"

def test_skip_propagation():
    """Test that descendants of skipped nodes are not executed"""
    graph = ExecutableGraph()
    
    condition_node = Node(
        name="condition",
        action_function=always_false
    )
    graph.add_node_to_graph(condition_node)
    
    true_node = Node(
        name="true_branch",
        action_function=true_branch,
        condition=true_condition
    )
    graph.add_node_to_graph(true_node)
    
    final_node = Node(
        name="final_node",
        action_function=final_node_action
    )
    graph.add_node_to_graph(final_node)
    
    graph.add_edge(condition_node, true_node, argument_name="x")
    graph.add_edge(true_node, final_node)
    
    graph.execute_graph()
    
    assert condition_node.is_completed()
    assert condition_node.output is False
    assert true_node.state == NodeState.SKIPPED
    assert true_node.output is None
    assert final_node.state == NodeState.SKIPPED
    assert final_node.output is None