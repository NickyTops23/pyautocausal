from typing import Callable, Dict, Optional
from pathlib import Path
from .graph import ExecutableGraph
from .nodes import Node, InputNode
from ..persistence.output_config import OutputConfig
from ..persistence.output_types import OutputType
from ..persistence.local_output_handler import LocalOutputHandler

class GraphBuilder:
    """Builder class for creating execution graphs with a fluent interface."""
    
    def __init__(self, output_path: Optional[Path] = None):
        self.graph = ExecutableGraph(
            output_handler=LocalOutputHandler(output_path) if output_path else None
        )
        self.nodes = {}
    
    def create_node(
        self,
        name: str,
        action_function: Callable,
        predecessors: Optional[Dict[str, str]] = None,
        condition: Optional[Callable[..., bool]] = None,
        skip_reason: str = "",
        output_config: Optional[OutputConfig] = None,
    ) -> "GraphBuilder":
        """
        Add a node to the graph.
        
        Args:
            name: Name of the node
            action_function: Function to execute
            predecessors: Dict mapping argument names to predecessor node names
            condition: Optional function that determines if node should execute
            skip_reason: Message explaining why node was skipped if condition is False
            output_config: Configuration for node output
            
        Returns:
            self for method chaining
        """
        # Create the node
        node = Node(
            name=name,
            action_function=action_function,
            condition=condition,
            skip_reason=skip_reason,
            output_config=output_config
        )

        node.set_graph(self.graph)
        
        # Store node reference
        self.nodes[name] = node
        
        # Add predecessors if specified
        if predecessors:
            for arg_name, pred_name in predecessors.items():
                if pred_name not in self.nodes:
                    raise ValueError(
                        f"Predecessor node '{pred_name}' not found for argument '{arg_name}'"
                    )
                node.add_predecessor(
                    self.nodes[pred_name],
                    argument_name=arg_name
                )
        
        return self
    
    def add_node(self, name: str, node: Node, predecessors: Optional[Dict[str, str]] = None) -> "GraphBuilder":
        """
        Add an existing node to the graph.
        
        Args:
            name: Name of the node
            node: Node to add
            
        Returns:
            self for method chaining
        """
        # Set graph to the builder's graph
        node.set_graph(self.graph)
        

        self.nodes[name] = node
        
        # Add predecessors if specified
        if predecessors:
            for arg_name, pred_name in predecessors.items():
                if pred_name not in self.nodes:
                    raise ValueError(
                        f"Predecessor node '{pred_name}' not found for argument '{arg_name}'"
                    )
                node.add_predecessor(
                    self.nodes[pred_name],
                    argument_name=arg_name
                )
        return self

    def add_input_node(self, name: str) -> "GraphBuilder":
        """
        Add an input node to the graph.
        
        Args:
            name: Name of the input node
            
        Returns:
            self for method chaining
        """
        node = InputNode(name=name, graph=self.graph)
        self.nodes[name] = node
        if not hasattr(self.graph, 'input_nodes'):
            self.graph.input_nodes = {}
        self.graph.input_nodes[name] = node
        return self
    


    def build(self) -> ExecutableGraph:
        """Build and return the configured graph."""
        return self.graph