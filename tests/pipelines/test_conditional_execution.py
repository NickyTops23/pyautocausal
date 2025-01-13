import pytest
from pyautocausal.orchestration.nodes import Node
from pyautocausal.orchestration.graph import ExecutableGraph

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

def test_true_condition():
    """Test that when condition is True, only true branch executes"""
    graph = ExecutableGraph()
    
    condition_node = Node("condition", graph, always_true)
    true_node = Node(
        "true_branch", 
        graph, 
        true_branch,
        condition=is_true,
        skip_reason="Condition was false"
    )
    false_node = Node(
        "false_branch", 
        graph, 
        false_branch,
        condition=is_false,
        skip_reason="Condition was true"
    )
    
    true_node.add_predecessor(condition_node, argument_name="x")
    false_node.add_predecessor(condition_node, argument_name="x")
    
    graph.execute_graph()
    
    assert condition_node.is_completed()
    assert condition_node.output is True
    assert true_node.is_completed()
    assert true_node.output == "true_branch_executed"
    assert false_node.is_completed()  # Completed but skipped
    assert false_node.output is None

def test_false_condition():
    """Test that when condition is False, only false branch executes"""
    graph = ExecutableGraph()
    
    condition_node = Node("condition", graph, always_false)
    true_node = Node(
        "true_branch", 
        graph, 
        true_branch,
        condition=is_true,
        skip_reason="Condition was false"
    )
    false_node = Node(
        "false_branch", 
        graph, 
        false_branch,
        condition=is_false,
        skip_reason="Condition was true"
    )
    
    true_node.add_predecessor(condition_node, argument_name="x")
    false_node.add_predecessor(condition_node, argument_name="x")
    
    graph.execute_graph()
    
    assert condition_node.is_completed()
    assert condition_node.output is False
    assert true_node.is_completed()  # Completed but skipped
    assert true_node.output is None
    assert false_node.is_completed()
    assert false_node.output == "false_branch_executed"