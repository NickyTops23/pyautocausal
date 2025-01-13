import networkx as nx
from typing import Callable, Any, Optional
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
    
class Node(BaseNode):
    def __init__(
            self, 
            name: str, 
            graph: DiGraph, 
            action_function: Callable,
            output_config: OutputConfig = None,
            condition: Callable[..., bool] = None,
            skip_reason: str = "",
            action_condition_kwarg_map: dict[str, str] = {},
        ):
        super().__init__(name, graph)
        self.state = NodeState.PENDING
        self.condition = condition
        self.skip_reason = skip_reason
        self.output = None
        self.output_config = output_config or OutputConfig()
        self.predecessor_outputs = {}
        self.action_condition_kwarg_map = action_condition_kwarg_map
        self.action_function = action_function
    
    def validate_condition(self):
        """Verify that condition is valid given the node's predecessors"""
        if not self.predecessor_outputs:
            raise ValueError("No predecessor outputs found. Please call get_predecessor_outputs() before calling validate_condition()")
        if self.condition is not None and not self.get_predecessors():
            raise ValueError(f"Node {self.name} has a condition but no predecessors")
        if self.condition is not None:
            # check that the keyword arguments of the condition function map to existing predecessors
            for arg in self.condition.__code__.co_varnames:
                mapped_arg = self.action_condition_kwarg_map.get(arg, arg)
                if mapped_arg not in self.predecessor_outputs:
                    raise ValueError(f"Condition function {self.condition.__name__} has argument {arg} which is not a predecessor")
    
    def should_execute(self) -> bool:
        if self.condition is not None:
            if not self.predecessor_outputs:
                self.get_predecessor_outputs()
            try:
                self.validate_condition()
                # Since validate_condition() ensures all required arguments exist,
                # we can pass predecessor_outputs directly after mapping
                mapped_predecessor_outputs = {self.action_condition_kwarg_map.get(arg, arg): value for arg, value in self.predecessor_outputs.items()}
                should_run = self.condition(**mapped_predecessor_outputs)
                
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
    
    def get_predecessor_outputs(self):
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
        self.predecessor_outputs = predecessor_outputs
        
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

    def _execute(self):
        """Execute the node's action function with predecessor outputs"""
        self.get_predecessor_outputs()
        
        # Check if action function accepts any parameters
        params = self.action_function.__code__.co_varnames[:self.action_function.__code__.co_argcount]
        if params:
            # Only pass predecessor outputs if the function expects parameters
            self.output = self.action_function(**self.predecessor_outputs)
        else:
            # Execute without parameters if function takes no arguments
            self.output = self.action_function()