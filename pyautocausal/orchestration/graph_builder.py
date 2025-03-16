from typing import Callable, Dict, Optional, Any
from pathlib import Path
from .graph import ExecutableGraph
from .nodes import Node, InputNode
from ..persistence.output_config import OutputConfig
from ..persistence.output_types import OutputType
from ..persistence.local_output_handler import LocalOutputHandler
from .condition import Condition

class GraphBuilder:
    """Builder class for creating execution graphs with a fluent interface."""
    
    def __init__(self, output_path: Optional[Path] = None):
        self.graph = ExecutableGraph(
            output_handler=LocalOutputHandler(output_path) if output_path else None
        )

    def create_node(
        self,
        name: str,
        action_function: Callable,
        predecessors: Optional[Dict[str, str]] = None,
        condition: Optional[Condition] = None,
        output_config: Optional[OutputConfig] = None,
        save_node: bool = False,
    ) -> "GraphBuilder":
        """
        Add a node to the graph.
        
        Args:
            name: Name of the node
            action_function: Function to execute
            predecessors: Dict mapping argument names to predecessor node names
            condition: Optional Condition object that determines if node should execute
            output_config: Optional configuration for node output
            save_node: Whether to save the node's output
            
        Returns:
            self for method chaining
        """
        # Create the node
        node = Node(
            name=name,
            action_function=action_function,
            condition=condition,
            output_config=output_config,
            save_node=save_node
        )
        
        # Use add_node to handle the rest
        return self.add_node(name, node, predecessors)
    
    def add_node(self, node: Node, predecessors: Optional[Dict[str, str]] = None) -> "GraphBuilder":
        """
        Add an existing node to the graph.
        
        Args:
            node: Node to add
            predecessors: Dict mapping argument names to predecessor node names
            
        Returns:
            self for method chaining
        """

        # Add node to graph
        self.graph.add_node_to_graph(node)

        # Add predecessors if specified
        if predecessors:
            for arg_name, pred_name in predecessors.items():
                # throws error if predecessor not found
                self.graph.add_edge(self.graph.get(pred_name), node, argument_name=arg_name)
        return self

    def add_input_node(self, name: str, input_dtype: type = Any) -> "GraphBuilder":
        """
        Add an input node to the graph.
        
        Args:
            name: Name of the input node
            dtype: Expected input type for this node
            
        Returns:
            self for method chaining
        """
        self.graph.add_input_node(name, input_dtype)
        return self
    
    def build(self) -> ExecutableGraph:
        """Build and return the configured graph."""
        return self.graph
    
