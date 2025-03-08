import networkx as nx
from typing import Set, Optional, Dict, Any
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
        self._nodes_by_name = {}  # New dictionary to track nodes by name
        
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

    def get(self, name: str) -> Node:
        """Get a node by its name.
        
        Args:
            name: Name of the node to find
            
        Returns:
            The node with the given name
            
        Raises:
            ValueError: If no node exists with the given name
        """
        node = self._nodes_by_name.get(name)
        if node is None:
            raise ValueError(f"No node found with name '{name}'")
        return node

    def add_node(self, node, **attr):
        """Override add_node to check for name conflicts and maintain name mapping."""
        if hasattr(node, 'name'):
            if node.name in self._nodes_by_name:
                raise ValueError(
                    f"Cannot add node: a node with name '{node.name}' already exists in the graph"
                )
            self._nodes_by_name[node.name] = node
        else:
            raise ValueError(f"Node must have a name, got {type(node)}")
        
        super().add_node(node, **attr)

    def add_input_node(self, name: str, input_dtype: type = Any):
        """Add an input node to the graph."""
        from .nodes import InputNode
        input_node = InputNode(name=name, graph=self, input_dtype=input_dtype)
        self._input_nodes[name] = input_node        

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

        targets = set()
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
            targets.add(target)
        
        # Create mapping of old nodes to new nodes
        node_mapping = {}
        
        # Re-initialize nodes from other graph
        for node in other.nodes():
            # Generate unique name if there's a conflict
            new_name = node.name
            counter = 1
            while new_name in self._nodes_by_name:
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
                if node in targets:
                    # Target nodes are no longer input nodes
                    # we need to make a new regular node that simply passes the input
                    def pass_input(x: node.input_dtype) -> node.input_dtype:
                        return x
                    
                    new_node = Node(
                        name=new_name,
                        action_function=pass_input,
                        graph=self,
                    )
                else:
                    self.add_input_node(new_name, node.input_dtype)
                node_mapping[node] = self.get(new_name)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")

            
            

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

    def to_text(self) -> str:
        """Generate a text representation of the graph for debugging.
        
        Returns:
            A string containing a text visualization of the graph structure.
        """
        from .nodes import InputNode
        output = []
        
        # Header
        output.append("Graph Structure:")
        output.append("=" * 50)
        
        # Nodes section
        output.append("\nNodes:")
        output.append("-" * 20)
        for node in sorted(self._nodes_by_name.values(), key=lambda n: n.name):
            node_type = "InputNode" if isinstance(node, InputNode) else "Node"
            state = f"[{node.state.value}]" if hasattr(node, "state") else ""
            output.append(f"{node.name} ({node_type}) {state}")
        
        # Edges section
        output.append("\nConnections:")
        output.append("-" * 20)
        for u, v, data in sorted(self.edges(data=True), key=lambda x: (x[0].name, x[1].name)):
            arg_name = f" as '{data.get('argument_name')}'" if data.get('argument_name') else ""
            output.append(f"{u.name} -> {v.name}{arg_name}")
        
        # Input nodes section
        output.append("\nExternal Inputs:")
        output.append("-" * 20)
        for name, node in sorted(self.input_nodes.items()):
            preds = list(self.predecessors(node))
            if not preds:  # Only show external inputs
                dtype = getattr(node, 'input_dtype', 'Any').__name__
                output.append(f"{name} (expects {dtype})")
        
        return "\n".join(output)

    def print_graph(self):
        """Print a text representation of the graph."""
        print(self.to_text()) 