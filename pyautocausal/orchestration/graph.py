import networkx as nx
from typing import Set, Optional
from .base import Node
from ..persistence.output_handler import OutputHandler
from pyautocausal.utils.logger import get_class_logger


class ExecutableGraph(nx.DiGraph):
    def __init__(self, output_handler: Optional[OutputHandler] = None):
        super().__init__()
        self.logger = get_class_logger(self.__class__.__name__)
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
    
    def is_execution_complete(self) -> bool:
        """Returns True if all nodes in the graph are completed"""
        return all(node.is_completed() 
                  for node in self.nodes() if isinstance(node, Node))

    def get_incomplete_nodes(self) -> Set[Node]:
        """Returns all nodes that haven't been completed yet"""
        return {node for node in self.nodes() 
               if isinstance(node, Node) and not node.is_completed()}
    
    def save_node_output(self, node: Node):
        """Save node output if configured to do so"""
        if (node.output_config.save_output and 
            node.output is not None):
            if self.save_node_outputs:
                output_name = getattr(node.output_config, 'output_name', None) or node.name
                if node.output_config.output_type is None:
                    raise ValueError(f"Output type is not set for node {node.name}")
                self.output_handler.save(output_name, node.output, node.output_config.output_type)
            else:
                self.logger.warning(f"Node {node.name} output not saved because save_node_outputs is False")
            
    def execute_graph(self):
        """Execute all nodes in the graph in dependency order"""
        while True:
            ready_nodes = self.get_ready_nodes()
            running_nodes = self.get_running_nodes()
            
            if not ready_nodes and not running_nodes:
                if self.is_execution_complete():
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