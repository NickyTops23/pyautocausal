import networkx as nx
from typing import Callable, Any, Optional, Union
from .node_state import NodeState
from .base import OutputConfig
from .graph import ExecutableGraph
from ..persistence.output_types import OutputType
from ..persistence.output_config import OutputConfig
from networkx import DiGraph
import inspect
from .condition import Condition
import pandas as pd
import matplotlib.pyplot as plt
from ..persistence.type_inference import infer_output_type
from ..persistence.serialization import prepare_output_for_saving

class BaseNode:
    def __init__(self, name: str, graph: Optional[ExecutableGraph] = None):
        self.name = name
        self.graph = None
        if graph is not None:
            self.set_graph(graph)
    
    def __str__(self):
        return f"BaseNode(name={self.name})"
    
    def set_graph(self, graph: ExecutableGraph):
        """Set the graph for this node and add the node to the graph."""
        self.graph = graph
        self.graph.add_node(self)
    
    def add_successor(self, successor: 'BaseNode'):
        if self.graph is None:
            raise ValueError("Node must be added to a graph before adding successors")
        self.graph.add_edge(self, successor)
        
    def add_predecessor(self, predecessor: 'BaseNode', argument_name: Optional[str] = None):
        if self.graph is None:
            raise ValueError("Node must be added to a graph before adding predecessors")
        self.graph.add_edge(predecessor, self, argument_name=argument_name)

    def __rshift__(self, other: 'BaseNode') -> 'BaseNode':
        """Implements the >> operator for node wiring with type validation"""
        if not isinstance(other, InputNode):
            raise ValueError(f"Right-hand node must be an input node, got {type(other)}")
        
        # Get return type from action function signature
        return_annotation = inspect.signature(self.action_function).return_annotation
        
        # Get expected input type from InputNode
        input_type = other.input_type
        
        # Warn if types cannot be validated
        if return_annotation == inspect.Parameter.empty:
            self.graph.logger.warning(
                f"Cannot validate connection: {self.name} -> {other.name}. "
                f"Node {self.name}'s action function lacks return type annotation."
            )
        elif input_type == Any:
            self.graph.logger.warning(
                f"Cannot validate connection: {self.name} -> {other.name}. "
                f"Input node {other.name} accepts Any type."
            )
        # Validate types if both are specified
        elif return_annotation != inspect.Parameter.empty and input_type != Any:
            if not issubclass(return_annotation, input_type):
                raise TypeError(
                    f"Type mismatch in connection {self.name} -> {other.name}: "
                    f"Node outputs {return_annotation.__name__}, but input node expects {input_type.__name__}"
                )
        
        self.add_successor(other)
        return other  # Return other to allow chaining

class Node(BaseNode):
    def __init__(
            self, 
            name: str, 
            action_function: Callable,
            graph: Optional[ExecutableGraph] = None,
            output_config: Optional[OutputConfig] = None,
            condition: Optional[Condition] = None,
            action_condition_kwarg_map: dict[str, str] = {},
            save_node: bool = False,
        ):
        super().__init__(name, graph)
        
        # Validate and setup output configuration
        return_annotation = self._get_return_annotation(action_function)
        self._validate_save_configuration(save_node, return_annotation, output_config)
        self.output_config = self._setup_output_config(save_node, return_annotation, output_config)
        
        # Initialize node state and attributes
        self.state = NodeState.PENDING
        self.condition = condition
        self.output = None
        self.predecessor_outputs = {}
        self.action_condition_kwarg_map = action_condition_kwarg_map
        self.action_function = action_function
        self.execution_count = 0
    
    def _get_return_annotation(self, action_function: Callable) -> Any:
        """Get the return type annotation from the action function"""
        signature = inspect.signature(action_function)
        return signature.return_annotation

    def _validate_save_configuration(
            self, 
            save_node: bool, 
            return_annotation: Any, 
            output_config: Optional[OutputConfig]
        ) -> None:
        """Validate the saving configuration and type annotations"""
        if save_node and return_annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Node {self.name}: When save_node=True, the action function must have a return type annotation. "
                "Either:\n"
                "1. Add a return type hint to your function:\n"
                "   def my_function() -> pd.DataFrame:\n"
                "       return pd.DataFrame(...)\n\n"
                "2. Or specify an output_config with an output_type:\n"
                "   output_config=OutputConfig(output_type=OutputType.PARQUET)"
            )
        
        if output_config is not None and not save_node:
            raise ValueError(
                f"Node {self.name}: Cannot specify output_config when save_node is False"
            )

    def _setup_output_config(
            self, 
            save_node: bool, 
            return_annotation: Any, 
            output_config: Optional[OutputConfig]
        ) -> Optional[OutputConfig]:
        """Setup the output configuration based on save settings and type annotation"""
        if output_config is None and save_node:
            output_type = infer_output_type(return_annotation)
            return OutputConfig(
                output_type=output_type,
                output_filename=self.name
            )
        return output_config
    
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
                self._resolve_function_arguments(self.condition.condition_func, mapped_outputs)
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
            name: None
            for name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty
        }
        missing_required = set(required_params.keys())

        optional_params = {
            name: None
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        missing_optional = set(optional_params.keys())
        
        # Start with available arguments
        arguments = available_args.copy()
        for argument in arguments.keys():
            if argument in required_params:
                required_params[argument] = arguments[argument]
                missing_required.remove(argument)
            elif argument in optional_params:
                optional_params[argument] = arguments[argument]
                missing_optional.remove(argument)
        
        # For any missing required parameters, try to get them from run context
        
        if missing_required and hasattr(self.graph, 'run_context'):
            for param in missing_required:
                if hasattr(self.graph.run_context, param):
                    required_params[param] = getattr(self.graph.run_context, param)
                    missing_required.remove(param)

        
        # Check if we have all required parameters
        if missing_required:
            raise ValueError(
                f"Missing required parameters for {func.__name__} in node {self.name}: "
                f"{missing_required}. Not found in available arguments or run context."
            )
        
        # For optional parameters, try to get them from available args or run context
        # but don't raise an error if they're not found

        keys_to_delete = []
        for param_name, current_value in optional_params.items():
            if param_name in missing_optional:
                # Try run context
                if hasattr(self.graph, 'run_context') and hasattr(self.graph.run_context, param_name):
                    optional_params[param_name] = getattr(self.graph.run_context, param_name)
                    missing_optional.remove(param_name)
                else: 
                    # delete key from optional_params to indicate it's not found
                    # this helps us know which default values we're overriding
                    keys_to_delete.append(param_name)
        
        for key in keys_to_delete:
            if key in optional_params:
                del optional_params[key]
        
        return {**required_params, **optional_params}

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
            arguments = self._resolve_function_arguments(self.condition.condition_func, mapped_outputs)
            return self.condition.evaluate(**arguments)
        
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
        self.execution_count += 1
        try:
            # Check condition satisfaction before executing
            condition_satisfied = self.condition_satisfied()
            has_skipped_predecessors = self.has_skipped_predecessors()
            
            if has_skipped_predecessors:
                self.mark_skipped()
                self.graph.logger.info(
                    f"Skipping {self.name}: predecessor nodes were skipped"
                )
                return
            
            if not condition_satisfied:
                self.mark_skipped()
                if self.condition:
                    self.graph.logger.info(
                        f"Skipping {self.name}: condition '{self.condition.description}' was not satisfied"
                    )
                return
            
            self.mark_running()
            if self.condition:
                self.graph.logger.info(
                    f"Executing {self.name}: condition '{self.condition.description}' was satisfied"
                )
            self._execute()
            self.mark_completed()
        except Exception as e:
            self.mark_failed()
            raise e

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
        """Execute the node's action function with predecessor outputs"""
        self.get_predecessor_outputs()
        arguments = self._resolve_function_arguments(self.action_function, self.predecessor_outputs)
        print(f"Executing {self.name} with arguments: {arguments}")
        output = (
            self.action_function(**arguments) if arguments 
            else self.action_function()
        )
        
        if self.output_config:
            self.output = prepare_output_for_saving(output, self.output_config.output_type)
        else:
            self.output = output

class InputNode(BaseNode):
    """A node that accepts external input and passes it to its successors."""
    
    def __init__(self, name: str, graph: ExecutableGraph, dtype: type = Any):
        super().__init__(name, graph)
        self.state = NodeState.PENDING
        self.output = None
        self.dtype = dtype
    
    def set_input(self, value: Any):
        """Set the input value that will be passed to successor nodes"""
        if self.dtype is not Any and not isinstance(value, self.dtype):
            raise TypeError(f"Input value for node '{self.name}' must be of type {self.dtype.__name__}, got {type(value).__name__}")
        self.output = value
        self.state = NodeState.COMPLETED

    def is_skipped(self) -> bool:
        return False # Input nodes are never skipped
    
    def is_ready(self) -> bool:
        return False  # Input nodes are never ready for execution
    
    def is_running(self) -> bool:
        return False
    
    def is_completed(self) -> bool:
        return self.state == NodeState.COMPLETED
    
    def execute(self) -> None:
        pass  # Input nodes don't execute; they just pass through their input


