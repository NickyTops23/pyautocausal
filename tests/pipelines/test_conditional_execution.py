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

# Function that would raise an error if called with None
def requires_attribute_access(df):
    # This will raise AttributeError if df is None
    return 'columns' in df

# Create reusable conditions
true_condition = Condition(is_true, "Condition is true")
false_condition = Condition(is_false, "Condition is false")
attribute_condition = Condition(requires_attribute_access, "Condition requires attribute access")

def test_true_condition():
    """Test that when condition is True, only true branch executes"""
    graph = ExecutableGraph()
    
    # Create nodes
    condition_node = Node(
        name="condition",
        action_function=always_true,
        graph=graph
    )
    
    true_node = Node(
        name="true_branch",
        action_function=true_branch,
        condition=true_condition,
        graph=graph
    )
    true_node.add_predecessor(condition_node, argument_name="x")
    
    false_node = Node(
        name="false_branch",
        action_function=false_branch,
        condition=false_condition,
        graph=graph
    )
    false_node.add_predecessor(condition_node, argument_name="x")
    
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
        action_function=always_false,
        graph=graph
    )
    
    true_node = Node(
        name="true_branch",
        action_function=true_branch,
        condition=true_condition,
        graph=graph
    )
    true_node.add_predecessor(condition_node, argument_name="x")
    
    false_node = Node(
        name="false_branch",
        action_function=false_branch,
        condition=false_condition,
        graph=graph
    )
    false_node.add_predecessor(condition_node, argument_name="x")
    
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
        action_function=always_false,
        graph=graph
    )
    
    true_node = Node(
        name="true_branch",
        action_function=true_branch,
        condition=true_condition,
        graph=graph
    )
    
    final_node = Node(
        name="final_node",
        action_function=final_node_action,
        graph=graph
    )
    
    true_node.add_predecessor(condition_node, argument_name="x")
    final_node.add_predecessor(true_node)
    
    graph.execute_graph()
    
    assert condition_node.is_completed()
    assert condition_node.output is False
    assert true_node.state == NodeState.SKIPPED
    assert true_node.output is None
    assert final_node.state == NodeState.SKIPPED
    assert final_node.output is None

def test_condition_not_evaluated_with_skipped_predecessors():
    """Test that conditions are not evaluated when predecessors are skipped.
    
    This test verifies the fix for the bug where conditions were being evaluated
    even when predecessors were skipped, which could lead to errors like
    'NoneType' object has no attribute 'columns'.
    """
    graph = ExecutableGraph()
    
    # First branch - will be skipped due to condition
    branch_condition = Node(
        name="branch_condition",
        action_function=always_false,  # This will make the next node skip
        graph=graph
    )
    
    skipped_node = Node(
        name="skipped_node",
        action_function=lambda: {"columns": ["a", "b", "c"]},  # Returns a dict with columns
        condition=true_condition,  # This condition won't be satisfied
        graph=graph
    )
    skipped_node.add_predecessor(branch_condition, argument_name="x")
    
    # This node would raise an error if it tried to evaluate its condition
    # with None input from the skipped predecessor
    attribute_dependent_node = Node(
        name="attribute_dependent_node",
        action_function=final_node_action,
        condition=attribute_condition,  # This would raise an error if evaluated with None
        graph=graph
    )
    attribute_dependent_node.add_predecessor(skipped_node, argument_name="df")
    
    # Execute the graph - this should not raise an error
    graph.execute_graph()
    
    # Verify that nodes were skipped correctly
    assert branch_condition.is_completed()
    assert branch_condition.output is False
    assert skipped_node.is_skipped()
    assert skipped_node.output is None
    assert attribute_dependent_node.is_skipped()
    assert attribute_dependent_node.output is None