from typing import Type, Optional, Union, Any
import pandas as pd
from .graph import ExecutableGraph
from .data_input import DataInput
from .nodes import Node
from ..persistence.output_handler import OutputHandler
from .run_context import RunContext

class PipelineGraph(ExecutableGraph):
    """Base class for pipeline-specific graphs"""
    
    # Class variable to be overridden by subclasses
    input_class: Type[DataInput] = None
    
    def __init__(
        self,
        output_handler: Optional[OutputHandler] = None,
        run_context: Optional[RunContext] = None
    ):
        if not self.input_class:
            raise ValueError(
                f"Pipeline graph {self.__class__.__name__} must define input_class"
            )
        
        super().__init__(
            output_handler=output_handler,
            run_context=run_context
        )
        
        self.starter_nodes = {}
        self.node_index = {}  

    def fit(self, input_data: Any):
        """Fit the pipeline graph to the input data
        
        Args:
            input_data: Input data that will be passed to the pipeline's input_class
                       constructor if not already an instance of input_class
        """
        # If input is not already wrapped in input_class, create new instance
        if not isinstance(input_data, self.input_class):
            try:
                input_data = self.input_class(input_data)
            except Exception as e:
                raise TypeError(
                    f"Could not create {self.input_class.__name__} from input data: {str(e)}"
                )

        # Check if all required fields are present
        self.input_class.check_presence_of_required_fields()
        
        #TODO: Remove any BaseNode

        # Add nodes from input data
        self.add_initial_nodes_from_dict(input_data.to_dict())
        
        # Execute the graph
        self.execute_graph()
        
        return self
        
    def add_node(self, node: Node):
        """Add a node to the graph and track it in the node index.
        
        Args:
            node: Node to add to the graph
            
        Raises:
            ValueError: If a node with the same name already exists
        """
        if node.name in self.node_index:
            raise ValueError(f"Node with name '{node.name}' already exists in the graph")
        
        super().add_node(node)
        self.node_index[node.name] = node

    def add_initial_nodes_from_dict(self, initial_node_dict: dict):
        """Create initial nodes from input dictionary"""
        
        for node_name, node_output in initial_node_dict.items():
            node = Node(
                name=node_name,
                graph=self, 
                action_function=lambda x=node_output: x
            )
            
            self.connect_initial_nodes(node)

    
    def connect_initial_nodes(self,initial_node: Node):
        """ Connect a node to the starter nodes """
        for node_name, starter_node in self.starter_nodes.items():
            starter_node.add_predecessor(initial_node, argument_name = "df")

    def add_branch(self, steps: list[tuple[str, Node]], predecessor: str = None):
        """Add a branch of nodes to the graph.
        
        Args:
            steps: List of (name, node) tuples defining the branch
            predecessor: Name of node to connect branch to (optional)
        """
        if not steps:
            return
        
        # Add first node and connect to predecessor if specified
        first_node_name, first_node = steps[0]
        first_node.name = first_node_name
        first_node.graph = self
        self.add_node(first_node)
        
        # Handle predecessor connection
        if predecessor:
            if predecessor not in self.node_index:
                raise ValueError(
                    f"Predecessor node '{predecessor}' not found in graph. "
                    f"Available nodes: {list(self.node_index.keys())}"
                )
            pred_node = self.node_index[predecessor]
            first_node.add_predecessor(pred_node, argument_name='df')
            print(f"Added node: {first_node.name}, with predecessor: {pred_node.name}")
        else:
            self.starter_nodes[first_node_name] = first_node
            print(f"Added node: {first_node.name}, with no predecessor")
        
        # Connect remaining nodes in sequence
        current_node = first_node
        for node_name, node in steps[1:]:
            node.name = node_name
            node.graph = self
            self.add_node(node)
            node.add_predecessor(current_node, argument_name='df')
            print(f"Added node: {node.name}, with predecessor: {current_node.name}")
            current_node = node

