import networkx as nx
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
from .node_state import NodeState
from .base import OutputConfig
from .graph import ExecutableGraph
from ..persistence.output_types import OutputType
from ..persistence.output_config import OutputConfig
from networkx import DiGraph

class BaseNode:
    def __init__(self, name: str, graph: ExecutableGraph):
        self.name = name
        self.graph = graph
        self.graph.add_node(self)
    
    def __str__(self):
        return f"BaseNode(name={self.name})"
    
class Node(BaseNode, ABC):
    def __init__(self, name: str, graph: DiGraph, output_config: OutputConfig = None):
        super().__init__(name, graph)
        self.state = NodeState.PENDING
        self.condition: tuple[Callable[..., bool], list[str]] | None = None
        self.skip_reason: str | None = None
        self.output = None
        self.output_config = output_config or OutputConfig()
    
    def set_condition(self, condition: Callable[..., bool], predecessor_names: list[str], skip_reason: str = None):
        """
        Set a condition that must be True for this node to execute
        Args:
            condition: Function that returns bool, taking predecessor outputs as named arguments
            predecessor_names: List of predecessor node names whose outputs should be passed to condition
            skip_reason: Optional reason for skipping if condition is False
        """
        self.condition = (condition, predecessor_names)
        self.skip_reason = skip_reason
    
    def should_execute(self) -> bool:
        if self.condition is not None:
            try:
                condition_func, predecessor_names = self.condition
                # Get outputs from specified predecessors
                predecessor_outputs = {
                    name: self.graph.nodes[name]['node'].output 
                    for name in predecessor_names
                }
                should_run = condition_func(**predecessor_outputs)
                if not should_run:
                    self.mark_completed()
                    if self.skip_reason:
                        print(f"Skipping {self.name}: {self.skip_reason}")
                    return False
            except Exception as e:
                self.mark_failed()
                raise ValueError(f"Error evaluating condition for node {self.name}: {str(e)}")
        return True

    def execute(self):
        """Template method that handles state management and conditional execution"""
        try:
            if not self.should_execute():
                return
            
            self.mark_running()
            self._execute()
            self.mark_completed()
        except Exception as e:
            self.mark_failed()
            raise e

    def add_successor(self, successor: BaseNode):
        self.graph.add_edge(self, successor)
        
    def add_predecessor(self, predecessor: BaseNode, argument_name: Optional[str] = None):
        self.graph.add_edge(predecessor, self, argument_name=argument_name)
        
    def get_predecessors(self):
        return set(self.graph.predecessors(self))
    
    def get_successors(self):
        return set(self.graph.successors(self))
    
    def get_predecessor_outputs(self) -> dict[str, Any]:
        """Get outputs from immediate predecessor nodes"""

        # key in dict is the argument name if provided, otherwise the node name
        predecessors = self.get_predecessors()
        predecessor_outputs = {}
        for predecessor in predecessors:
            edge = self.graph.edges[predecessor, self]
            argument_name = edge.get('argument_name')
            if argument_name:
                predecessor_outputs[argument_name] = predecessor.output
            else:
                predecessor_outputs[predecessor.name] = predecessor.output
        return predecessor_outputs
        
    def mark_running(self):
        self.state = NodeState.RUNNING
        
    def mark_completed(self):
        self.state = NodeState.COMPLETED
        
    def mark_failed(self):
        self.state = NodeState.FAILED
        
    def is_completed(self):
        return self.state == NodeState.COMPLETED
    
    def is_running(self):
        return self.state == NodeState.RUNNING
    
    def is_ready(self) -> bool:
        """Returns True if all ancestors are completed and this node is pending"""
        if self.state != NodeState.PENDING:
            return False
        if not self.get_predecessors():
            return True
        return all(predecessor.is_completed() for predecessor in self.get_predecessors())

    @abstractmethod
    def _execute(self):
        """All nodes must implement an _execute method"""
        pass

class ActionNode(Node):
    def __init__(self, 
                 name: str, 
                 graph: nx.Graph, 
                 action_function: Callable,
                 output_config: OutputConfig = None):
        super().__init__(name, graph, output_config)
        self.action_function = action_function

    def _execute(self):
        inputs = self.get_predecessor_outputs()
        self.output = self.action_function(**inputs)
        self.mark_completed()