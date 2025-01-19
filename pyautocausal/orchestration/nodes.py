import networkx as nx
from typing import Callable, Any, Optional
from .node_state import NodeState
from .base import OutputConfig
from .graph import ExecutableGraph
from ..persistence.output_types import OutputType
from ..persistence.output_config import OutputConfig
from networkx import DiGraph
import inspect

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
            # Map predecessor outputs according to the condition argument mapping
            mapped_outputs = {
                self.action_condition_kwarg_map.get(arg, arg): value 
                for arg, value in self.predecessor_outputs.items()
            }
            
            # Use _resolve_function_arguments to check that all required arguments are available
            try:
                self._resolve_function_arguments(self.condition, mapped_outputs)
            except ValueError as e:
                raise ValueError(f"Invalid condition for node {self.name}: {str(e)}")
    
    def _resolve_function_arguments(self, func: Callable, available_args: dict) -> dict:
        """
        Resolve arguments for a function from available arguments and run context.
        Handles functions with default argument values.
        
        Args:
            func: The function that needs arguments
            available_args: Dictionary of already available arguments
            
        Returns:
            dict: Complete dictionary of resolved arguments
        """
        
        
        # Get information about function parameters
        signature = inspect.signature(func)
        required_params = {
            name: param 
            for name, param in signature.parameters.items()
            if param.default == inspect.Parameter.empty
        }
        
        # Start with available arguments
        arguments = available_args.copy()
        
        # For any missing required parameters, try to get them from run context
        missing_required = set(required_params.keys()) - set(arguments.keys())
        if missing_required and hasattr(self.graph, 'run_context'):
            for param in missing_required:
                if hasattr(self.graph.run_context, param):
                    arguments[param] = getattr(self.graph.run_context, param)
        
        # Check if we have all required parameters
        still_missing = set(required_params.keys()) - set(arguments.keys())
        if still_missing:
            raise ValueError(
                f"Missing required parameters for {func.__name__} in node {self.name}: "
                f"{still_missing}. Not found in available arguments or run context."
            )
        
        # For optional parameters, try to get them from available args or run context
        # but don't raise an error if they're not found
        optional_params = {
            name: param.default 
            for name, param in signature.parameters.items()
            if param.default != inspect.Parameter.empty
        }
        
        for param_name, default_value in optional_params.items():
            if param_name not in arguments:
                # Try run context first
                if hasattr(self.graph, 'run_context') and hasattr(self.graph.run_context, param_name):
                    arguments[param_name] = getattr(self.graph.run_context, param_name)
                # Otherwise use the default value
                else:
                    arguments[param_name] = default_value
        
        return arguments

    def condition_satisfied(self) -> bool:
        """Check if the node's condition is satisfied using predecessors and run context"""
        if self.condition is None:
            return True
        
        if not self.predecessor_outputs:
            self.get_predecessor_outputs()
        
        try:
            # Map predecessor outputs according to the condition argument mapping
            mapped_outputs = {
                arg: self.predecessor_outputs[self.action_condition_kwarg_map.get(arg, arg)]
                for arg in self.predecessor_outputs
            }
            
            # Resolve arguments including run context
            arguments = self._resolve_function_arguments(self.condition, mapped_outputs)
            return self.condition(**arguments)
        
        except Exception as e:
            raise ValueError(f"Error evaluating condition for node {self.name}: {str(e)}")

    def should_execute(self) -> bool:
        try:
            return self.condition_satisfied() and not self.has_skipped_predecessors()
        except Exception as e:
            self.mark_failed()
            raise ValueError(f"Error evaluating condition for node {self.name}: {str(e)}")

    def execute(self):
        """Template method that handles state management and conditional execution"""
        try:
            if not self.should_execute():
                self.mark_skipped()
                if self.skip_reason:
                    print(f"Skipping {self.name}: {self.skip_reason}")
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
    
    def mark_skipped(self):
        self.state = NodeState.SKIPPED
        
    def is_completed(self):
        return self.state == NodeState.COMPLETED
    
    def is_running(self):
        return self.state == NodeState.RUNNING
    
    def is_skipped(self):
        return self.state == NodeState.SKIPPED
    
    def has_skipped_predecessors(self):
        return any(predecessor.is_skipped() for predecessor in self.get_predecessors())
    
    def is_ready(self) -> bool:
        """Returns True if all ancestors are completed and this node is pending"""
        if self.state != NodeState.PENDING:
            return False
        if not self.get_predecessors():
            return True
        return all(predecessor.is_completed() or predecessor.is_skipped() for predecessor in self.get_predecessors())

    def _execute(self):
        """Execute the node's action function with predecessor outputs and run context"""
        self.get_predecessor_outputs()
        arguments = self._resolve_function_arguments(self.action_function, self.predecessor_outputs)
        self.output = (
            self.action_function(**arguments) if arguments 
            else self.action_function()
        )


