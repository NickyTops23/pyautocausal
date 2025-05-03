import networkx as nx
from typing import Callable, Any, Optional
from .node_state import NodeState
from .base import OutputConfig
from .graph import ExecutableGraph
from .result import Result
from ..persistence.output_types import OutputType
from ..persistence.output_config import OutputConfig
from networkx import DiGraph
from .result import Result
import inspect
from ..persistence.type_inference import infer_output_type
from ..persistence.serialization import prepare_output_for_saving
from pyautocausal.utils.logger import get_class_logger
import warnings
import types

class BaseNode:
    def __init__(self, name: str):
        self.name = name
        self.graph = None
    
    def __str__(self):
        return f"BaseNode(name={self.name})"
    
    # Remove the set_graph method that adds the node to the graph
    # Instead, we'll have a method that only sets the graph reference
    def _set_graph_reference(self, graph: ExecutableGraph):
        """Set the graph reference for this node. Should only be called by the graph."""
        if self.graph is not None and self.graph != graph:
            raise ValueError(f"Node {self.name} is already part of a different graph")
        self.graph = graph
    
class Node(BaseNode):
    def __init__(
            self, 
            name: str, 
            action_function: Callable,
            output_config: Optional[OutputConfig] = None,
            save_node: bool = False,
            node_description: str | None = None,
            notebook_function: Optional[Callable] = None
        ):
        super().__init__(name)
        self.logger = get_class_logger(f"{self.__class__.__name__}_{name}")
        
        # Validate and setup output configuration
        return_annotation = self._get_return_annotation(action_function)
        self._validate_save_configuration(save_node, return_annotation, output_config)
        self.output_config = self._setup_output_config(save_node, return_annotation, output_config)
        
        # Initialize node state and attributes
        self.state = NodeState.PENDING
        self.output = Result()
        self.output_to_save = None
        self.predecessor_outputs = {}
        self.action_function = action_function
        self.execution_count = 0
        self.node_description = node_description
        self.notebook_function = notebook_function
    
    def _get_return_annotation(self, action_function: Callable) -> Any:
        """Get the return type annotation from the action function"""
        signature = inspect.signature(action_function)
        return signature.return_annotation
    
    def __rshift__(self, other: 'BaseNode') -> tuple[BaseNode, BaseNode]:
        """Implements the >> operator for node wiring with type validation
        NB: This method is only part of the Node class because the >> syntax is cool.
        Args:
            other: The node to wire to
            
        Returns:
            A tuple containing the source and target nodes
            
        
        """
        if self.graph is None:
            raise ValueError("Node must be added to a graph before wiring")
        self.graph.can_wire_nodes(self, other)
        return (self, other)
    
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

    def _resolve_function_arguments(self, func: Callable, available_args: dict) -> dict:
        """
        Resolve arguments for a function from available arguments and run context.
        Handles functions with default argument values.
        We allow for lambda functions to take any argument as long as it's a single argument.
        
        Args:
            func: The function that needs arguments
            available_args: Dictionary of already available arguments
            
        Returns:
            dict: Complete dictionary of resolved arguments
        """
        
        is_lambda_with_single_argument = isinstance(func, types.LambdaType) and func.__code__.co_argcount == 1
        if is_lambda_with_single_argument:
            self.logger.info(f"Node {self.name}: has a lambda function with a single argument as either an action function or a condition function")
            available_args_keys = list(available_args.keys())
            if len(available_args_keys) == 1:
                self.logger.info(f"Node {self.name}: Lambda function with single argument {func.__name__} will use argument {available_args_keys[0]} from available arguments")
                return {func.__code__.co_varnames[0]: available_args[available_args_keys[0]]}
            else:
                raise ValueError(f"Lambda function with single argument {func.__name__} must take a single argument, got {list(available_args.keys())}")
        
        
        # Get information about function parameters
        signature = inspect.signature(func)
        required_params = {
            name: None
            for name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty # only include required parameters
              and param.kind != inspect.Parameter.VAR_KEYWORD # exclude varargs
        }
        missing_required = set(required_params.keys())

        optional_params = {
            name: None
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        has_varkeyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        # check if any keyword arguments are VAR_KEYWORD

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
            elif has_varkeyword:
                # if there are varargs we always add the argument to optional_params
                optional_params[argument] = arguments[argument]
        
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

    def execute(self):
        """Execute the node's action function"""
        self.execution_count += 1
        try:
            self.mark_running()
            self._execute()
            self.mark_completed()
        except Exception as e:
            self.mark_failed()
            raise e

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

    def get_result_value(self) -> Any:
        """Return the output value of this node from the result dictionary.
        
        This is a convenience method that returns the node's output value
        directly from the result_dict using the node's name as the key.
        
        Returns:
            The output value of this node
        """
        if self.output is None or self.output.result_dict is None:
            return None
        return self.output.result_dict.get(self.name)

    def _execute_action_function(self):
            # Get predecessor outputs directly from the graph
        predecessor_outputs = self.graph.get_node_predecessor_outputs(self)
        
        arguments = self._resolve_function_arguments(self.action_function, predecessor_outputs)
        print(f"Executing {self.name} with arguments: {arguments}")
        return (
            self.action_function(**arguments) if arguments 
            else self.action_function()
        )

    def _execute(self):
        """Execute the node's action function with predecessor outputs"""
        output = self._execute_action_function()
        
        self.output.update({self.name: output})
        if self.output_config:
            self.output_to_save = prepare_output_for_saving(output, self.output_config.output_type)
            
    
class DecisionNode(Node):
    """
    A node that makes a decision based on a condition and controls flow to downstream nodes.
    
    DecisionNode evaluates its condition and determines which downstream paths
    should be traversable. Nodes can be designated as "execute when true" (EWT)
    or "execute when false" (EWF).
    """
    def __init__(
            self, 
            name: str,
            condition: Callable,
            node_description: str | None = None
        ):
        # Create a passthrough function that aggregates predecessor outputs
        # by unioning the results of all predecessors
        def passthrough_function(**kwargs):
            result = {}
            for key, value in kwargs.items():
                result.update({key: value})
            return result
        
        super().__init__(
            name=name,
            action_function=passthrough_function,
            output_config=None,
            save_node=False,
            node_description=node_description
        )
        self.condition = condition
        # warn if the condition does not return a boolean
        if not callable(condition):
            raise ValueError(f"Condition for decision node {self.name} is not a callable function")
        # get the signature of the condition
        self.condition_signature = inspect.signature(condition)
        
        # warn if the condition does not have a return type annotation or returns a non-boolean value
        if self.condition_signature.return_annotation is inspect.Parameter.empty:
            warnings.warn(f"Condition for decision node {self.name} does not have a return type annotation")
        elif self.condition_signature.return_annotation != bool:
            warnings.warn(f"Condition for decision node {self.name} returns a non-boolean value: {self.condition_signature.return_annotation}")


        self._ewt_nodes = set()  # Nodes to execute when condition is True
        self._ewf_nodes = set()  # Nodes to execute when condition is False
    
    def add_execute_when_true(self, node):
        """Add a node to execute when condition evaluates to True"""
        self._ewt_nodes.add(node)
        return self
    
    def add_execute_when_false(self, node):
        """Add a node to execute when condition evaluates to False"""
        self._ewf_nodes.add(node)
        return self
    
    def validate_branches(self):
        """
        Validate that all successors have been classified as either EWT or EWF.
        This should be called before execution.
        """
        if not hasattr(self, 'graph') or self.graph is None:
            return  # Can't validate without a graph
            
        all_successors = set(self.graph.successors(self))

        # validate that successors are not classified as both EWT and EWF
        if self._ewt_nodes.intersection(self._ewf_nodes):
            raise ValueError(
                f"Decision node '{self.name}' has successors that are classified as both execute_when_true and execute_when_false: "
                f"{[node.name for node in self._ewt_nodes.intersection(self._ewf_nodes)]}"
            )

        all_classified = self._ewt_nodes.union(self._ewf_nodes)
        
        missing = all_successors - all_classified
        if missing:
            raise ValueError(
                f"Decision node '{self.name}' has successor nodes that are not classified as "
                f"execute_when_true or execute_when_false: {[node.name for node in missing]}"
            )
        
        extra = all_classified - all_successors
        if extra:
            raise ValueError(
                f"Decision node '{self.name}' has nodes classified as execute_when_true or "
                f"execute_when_false that are not successors: {[node.name for node in extra]}"
            )
    
    def condition_satisfied(self) -> bool:
        """Check if the node's condition is satisfied using predecessors and run context"""
        # Get predecessor outputs directly from the graph
        predecessor_outputs = self.graph.get_node_predecessor_outputs(self)
        
        try:
            # Resolve arguments including run context
            arguments = self._resolve_function_arguments(self.condition, predecessor_outputs)
            condition_result = self.condition(**arguments)
            if not isinstance(condition_result, bool):
                raise ValueError(f"Condition for node {self.name} returned a non-boolean value: {condition_result}")
            return condition_result
        
        except Exception as e:
            raise ValueError(f"Error evaluating condition for node {self.name}: {str(e)}")
    
    def _execute(self):
        """Execute the decision logic and control flow to downstream nodes"""
        

        # Validate that all successors are classified
        self.validate_branches()

        # executing the passthrough function merges all predecessor outputs into a single result
        self.output = Result(super()._execute_action_function())
        
        # Evaluate the condition
        condition_result = self.condition_satisfied()

        # Log the decision outcome
        self.logger.info(
            f"Decision node '{self.name}': condition evaluated to {condition_result}"
        )
        
        # Set traversability based on condition outcome, 
        for successor in self.graph.successors(self):
            edge = self.graph.edges[self, successor]
            
            if condition_result:
                if successor in self._ewt_nodes:
                    edge['traversable'] = True # this is not strictly necessary, since edges are traversable by default
                    self.logger.info(f"Edge to '{successor.name}' is traversable (condition is TRUE)")
                else:
                    edge['traversable'] = False
                    self.logger.info(f"Edge to '{successor.name}' is NOT traversable (condition is TRUE)")
            elif not condition_result:
                if successor in self._ewf_nodes:
                    edge['traversable'] = True
                    self.logger.info(f"Edge to '{successor.name}' is traversable (condition is FALSE)")
                else:
                    edge['traversable'] = False
                    self.logger.info(f"Edge to '{successor.name}' is NOT traversable (condition is FALSE)")
            else:
                raise ValueError(f"Decision node '{self.name}' received an unexpected condition result: {condition_result}")

class InputNode(Node):
    """A node that accepts external input and passes it to its successors."""
    
    def __init__(self, name: str, input_dtype: type = Any):
        self.input_dtype = input_dtype
        super().__init__(name, action_function=lambda: None)
        self.input_set = False
    
    def set_input(self, value: Any):
        """Set the input value that will be passed to successor nodes"""
        if self.input_set:
            raise ValueError(f"Input value for node '{self.name}' has already been set")
        
        if self.input_dtype is not Any and not isinstance(value, self.input_dtype):
            raise TypeError(f"Input value for node '{self.name}' must be of type {self.input_dtype.__name__}, got {type(value).__name__}")
        
        # set the action function in the superclass
        self.action_function = lambda: value
        self.input_set = True

    def _execute(self):
        if not self.input_set:
            raise ValueError(f"Input value for node '{self.name}' has not been set")
        super()._execute()


