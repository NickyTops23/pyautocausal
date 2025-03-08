import networkx as nx
from typing import Set, Optional, Dict
from .base import Node

from ..persistence.output_handler import OutputHandler
from pyautocausal.utils.logger import get_class_logger
from pyautocausal.orchestration.run_context import RunContext
from pyautocausal.orchestration.node_state import NodeState


class ExecutableGraph(nx.DiGraph):
    def __init__(
            self,
            output_handler: Optional[OutputHandler] = None,
            run_context: Optional[RunContext] = None
        ):
        super().__init__()
        self.logger = get_class_logger(self.__class__.__name__)
        self.run_context = run_context or RunContext()
        self._input_nodes = {}
        
        if output_handler is None:
            self.save_node_outputs = False
            self.logger.warning("No output handler provided, node outputs will not be saved")
        else:
            self.save_node_outputs = True
            self.output_handler = output_handler

    @property
    def input_nodes(self) -> Dict[str, 'InputNode']:
        """Get dictionary of input nodes"""
        return self._input_nodes

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
        from .nodes import InputNode  # Import here to avoid circular import
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
        # Get only input nodes without predecessors
        external_inputs = {
            name: node 
            for name, node in self.input_nodes.items() 
            if not list(self.predecessors(node))
        }
        
        # Validate inputs
        missing_inputs = set(external_inputs.keys()) - set(kwargs.keys())
        if missing_inputs:
            raise ValueError(f"Missing values for input nodes: {missing_inputs}")
        
        extra_inputs = set(kwargs.keys()) - set(external_inputs.keys())
        if extra_inputs:
            raise ValueError(f"Received values for non-existent input nodes: {extra_inputs}")
        
        # Validate and set input values
        for name, value in kwargs.items():
            input_node = external_inputs[name]
            input_node.set_input(value)  # Type checking happens in set_input
        
        # Execute the graph
        self.execute_graph()
        return self

    def add_node(self, node, **attr):
        """Override add_node to check for name conflicts."""
        if hasattr(node, 'name'):
            # Check for existing nodes with the same name
            existing_node = next(
                (n for n in self.nodes() if hasattr(n, 'name') and n.name == node.name),
                None
            )
            if existing_node is not None:
                raise ValueError(
                    f"Cannot add node: a node with name '{node.name}' already exists in the graph"
                )
        
        super().add_node(node, **attr)

    def add_input_node(self, name: str, node: 'InputNode'):
        """Add an input node to the graph."""
        from .nodes import InputNode
        if name in self._input_nodes:
            raise ValueError(f"Input node with name '{name}' already exists")
        if not isinstance(node, InputNode):
            raise ValueError(f"Node must be an InputNode, got {type(node)}")
        self._input_nodes[name] = node

    def merge_with(self, other: 'ExecutableGraph', *wirings) -> 'ExecutableGraph':
        """Merge another graph into this one with explicit wiring.
        
        Args:
            other: The graph to merge into this one
            *wirings: Variable number of wiring tuples created by the >> operator
            
        Example:
            g1.merge_with(g2, 
                node1 >> input_node1,
                node2 >> input_node2
            )
        """
        from .nodes import InputNode, Node as NodeObject

        # Validate states first
        non_pending_nodes = [
            node.name for node in self.nodes() 
            if hasattr(node, 'state') and node.state != NodeState.PENDING
        ]
        if non_pending_nodes:
            raise ValueError(
                "Cannot merge graphs: the following nodes in the target graph "
                f"are not in PENDING state: {non_pending_nodes}"
            )

        # Validate wirings
        if not wirings:
            raise ValueError(
                "At least one wiring (e.g., node1 >> input_node2) must be provided "
                "to ensure graphs are connected"
            )

        # Validate that all wirings are between the two graphs
        for wiring in wirings:
            source, target = wiring
            if source.graph is other and target.graph is self:
                raise ValueError(
                    f"Invalid wiring direction: {source.name} >> {target.name}. "
                    "Source must be from the target graph (first argument)"
                )
            if source.graph is not self or target.graph is not other:
                raise ValueError(
                    f"Invalid wiring: {source.name} >> {target.name}. "
                    "Source must be from the target graph and target from the source graph"
                )
            if not isinstance(target, InputNode):
                raise ValueError(
                    f"Invalid wiring: {source.name} >> {target.name}. "
                    "Target must be an InputNode"
                )

        # Get existing node names in self
        existing_names = {node.name for node in self.nodes()}
        
        # Create mapping of old nodes to new nodes
        node_mapping = {}
        
        # Re-initialize nodes from other graph
        for node in other.nodes():
            # Generate unique name if there's a conflict
            new_name = node.name
            counter = 1
            while new_name in existing_names:
                new_name = f"{node.name}_{counter}"
                counter += 1
            
            if new_name != node.name:
                self.logger.info(
                    f"Renaming node '{node.name}' to '{new_name}' during merge to avoid name conflict"
                )
            
            if isinstance(node, Node):
                new_node = NodeObject(
                    name=new_name,
                    action_function=node.action_function,
                    graph=self,
                    output_config=node.output_config,
                    condition=node.condition,
                    action_condition_kwarg_map=node.action_condition_kwarg_map,
                    save_node=bool(node.output_config)
                )
                node_mapping[node] = new_node
            elif isinstance(node, InputNode):
                new_node = InputNode(
                    name=new_name,
                    graph=self,
                    input_dtype=node.input_dtype
                )
                self.add_input_node(new_name, new_node)
                node_mapping[node] = new_node
            
            existing_names.add(new_name)

        # Add edges from the original graph
        for u, v, data in other.edges(data=True):
            if u in node_mapping and v in node_mapping:  # Only add edges between nodes we've mapped
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                self.add_edge(new_u, new_v, **data)

        # Add the wiring edges
        for wiring in wirings:
            source, target = wiring
            new_target = node_mapping[target]
            self.add_edge(source, new_target)

        return self 