import networkx as nx
from typing import Set, Optional
from .base import Node
from ..persistence.output_handler import OutputHandler
from pyautocausal.utils.logger import get_class_logger
from pyautocausal.orchestration.run_context import RunContext


class ExecutableGraph(nx.DiGraph):
    def __init__(
            self,
            output_handler: Optional[OutputHandler] = None,
            run_context: Optional[RunContext] = None
        ):
        super().__init__()
        self.logger = get_class_logger(self.__class__.__name__)
        self.run_context = run_context or RunContext()
        
        if output_handler is None:
            self.save_node_outputs = False
            self.logger.warning("No output handler provided, node outputs will not be saved")
        else:
            self.save_node_outputs = True
            self.output_handler = output_handler

    def get_ready_nodes(self) -> Set[Node]:
        """Returns all nodes that are ready to be executed"""
        return {node for node in self.nodes() 
               if isinstance(node, Node) and node.is_ready()}
    
    def get_running_nodes(self) -> Set[Node]:
        """Returns all nodes that are currently running"""
        return {node for node in self.nodes() 
               if isinstance(node, Node) and node.is_running()}
    
    def is_execution_finished(self) -> bool:
        """Returns True if all nodes in the graph are completed"""
        return self.get_incomplete_nodes() == set()

    def get_incomplete_nodes(self) -> Set[Node]:
        """Returns all nodes that haven't reached a terminal state yet"""
        return {node for node in self.nodes() 
                if isinstance(node, Node) and not node.state.is_terminal()}
    
    def save_node_output(self, node: Node):
        """Save node output if configured to do so"""
        from pyautocausal.orchestration.nodes import InputNode # Circular import
        if self.save_node_outputs & ~isinstance(node, InputNode): # Input nodes are not saved
            if (
                getattr(node, 'output_config', None) is not None
                and node.output is not None
            ):
                output_filename = getattr(node.output_config, 'output_filename', node.name)
                if node.output_config.output_type is None:
                    raise ValueError(f"Output type is not set for node {node.name}")
                self.output_handler.save(output_filename, node.output, node.output_config.output_type)
            else:
                self.logger.warning(f"Node {node.name} output not saved because no output config was provided")
            
    def execute_graph(self):
        """Execute all nodes in the graph in dependency order"""
        while True:
            ready_nodes = self.get_ready_nodes()
            running_nodes = self.get_running_nodes()
            
            if not ready_nodes and not running_nodes:
                if self.is_execution_finished():
                    break
                else:
                    incomplete = self.get_incomplete_nodes()
                    raise ValueError(
                        f"Graph execution stuck with incomplete nodes: "
                        f"{[node.name for node in incomplete]}"
                    )
            
            for node in ready_nodes:
                node.execute()
                self.save_node_output(node)  # Save output immediately after execution 

    def fit(self, **kwargs):
        """
        Set input values and execute the graph.
        
        Args:
            **kwargs: Dictionary mapping input node names to their values
        """
        if not hasattr(self, 'input_nodes'):
            raise ValueError("Graph has no input nodes. Add input nodes using GraphBuilder.add_input_node()")
        
        # Validate inputs
        missing_inputs = set(self.input_nodes.keys()) - set(kwargs.keys())
        if missing_inputs:
            raise ValueError(f"Missing values for input nodes: {missing_inputs}")
        
        extra_inputs = set(kwargs.keys()) - set(self.input_nodes.keys())
        if extra_inputs:
            raise ValueError(f"Received values for non-existent input nodes: {extra_inputs}")
        
        # Validate and set input values
        for name, value in kwargs.items():
            input_node = self.input_nodes[name]
            input_node.set_input(value)  # Type checking happens in set_input
        
        # Execute the graph
        self.execute_graph()
        return self 