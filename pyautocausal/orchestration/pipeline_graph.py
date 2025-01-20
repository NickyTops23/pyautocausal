from typing import Type, Optional
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
        input_data: DataInput,
        output_handler: Optional[OutputHandler] = None,
        run_context: Optional[RunContext] = None
    ):
        if not self.input_class:
            raise ValueError(
                f"Pipeline graph {self.__class__.__name__} must define input_class"
            )
            
        if not isinstance(input_data, self.input_class):
            raise TypeError(
                f"Input data must be instance of {self.input_class.__name__}, "
                f"got {type(input_data).__name__}"
            )
        
        # Check if all required fields are present in input dictionary
        self.input_class.check_presence_of_required_fields()
        
        super().__init__(
            output_handler=output_handler,
            run_context=run_context
        )
        
        self.starter_nodes = {}
        self.add_initial_nodes_from_dict(input_data.to_dict())
    
    def add_initial_nodes_from_dict(self, initial_node_dict: dict):
        """Create initial nodes from input dictionary"""
        for node_name, node_output in initial_node_dict.items():
            node = Node(
                name=node_name, 
                graph=self, 
                action_function=lambda x=node_output: x
            )
            self.add_node(node)
            self.starter_nodes[node_name] = node