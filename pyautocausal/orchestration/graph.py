import networkx as nx
from typing import Set, Optional, Dict, Any
from .base import Node

from ..persistence.output_handler import OutputHandler
from pyautocausal.utils.logger import get_class_logger
from pyautocausal.orchestration.run_context import RunContext
from pyautocausal.orchestration.node_state import NodeState
from inspect import Parameter, Signature


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
               if isinstance(node, Node) and self.is_node_ready(node)}
    
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
                if self.node_has_skipped_predecessors(node):
                    node.mark_skipped()
                    self.logger.info(
                        f"Skipping {node.name}: predecessor nodes were skipped"
                    )
                else:
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

    def add_node_to_graph(self, node, **attr):
        """Add a node to the graph and set the graph reference on the node."""
        if hasattr(node, 'name'):
            if node.name in self._nodes_by_name:
                raise ValueError(
                    f"Cannot add node: a node with name '{node.name}' already exists in the graph"
                )
            
            # Check if node is already in another graph
            if hasattr(node, 'graph') and node.graph is not None and node.graph != self:
                raise ValueError(
                    f"Cannot add node '{node.name}': node is already part of a different graph"
                )
            
            self._nodes_by_name[node.name] = node
            
            # Set the graph reference on the node
            node._set_graph_reference(self)
            
            # Add the node to the graph
            super().add_node(node, **attr)
        else:
            raise ValueError(f"Node must have a name, got {type(node)}")

    # Override the original add_node to prevent direct addition
    def add_node(self, node, **attr):
        """Override to ensure nodes are added through add_node_to_graph."""
        if hasattr(node, '_set_graph_reference'):
            # This is a BaseNode or subclass, so redirect to add_node_to_graph
            return self.add_node_to_graph(node, **attr)
        else:
            # This is not a BaseNode, so proceed with normal NetworkX behavior
            if hasattr(node, 'name'):
                self._nodes_by_name[node.name] = node
            super().add_node(node, **attr)

    def add_input_node(self, name: str, input_dtype: type = Any):
        """Add an input node to the graph."""
        from .nodes import InputNode
        input_node = InputNode(name=name, input_dtype=input_dtype)
        self.add_node_to_graph(input_node)
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
        non_pending_nodes_self = [
            node.name for node in self.nodes() 
            if hasattr(node, 'state') and node.state != NodeState.PENDING
        ]

        non_pending_nodes_other = [             
            node.name for node in other.nodes() 
            if hasattr(node, 'state') and node.state != NodeState.PENDING
        ]
        if non_pending_nodes_self:
            raise ValueError(
                "Cannot merge graphs: the following nodes in the source graph "
                f"are not in PENDING state: {non_pending_nodes_self}"
            )
        if non_pending_nodes_other:
            raise ValueError(
                "Cannot merge graphs: the following nodes in the target graph "
                f"are not in PENDING state: {non_pending_nodes_other}"
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
            if source.graph is not self or target.graph is not other:
                raise ValueError(
                    f"Invalid wiring: {source.name} >> {target.name}. "
                    "Source must be from the left graph and target from the right graph"
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
                    output_config=node.output_config,
                    condition=node.condition,
                    action_condition_kwarg_map=node.action_condition_kwarg_map,
                    save_node=bool(node.output_config)
                )
                self.add_node_to_graph(new_node)
                node_mapping[node] = new_node
            elif isinstance(node, InputNode):
                if node in targets:
                    def make_pass_function(dtype, param_name):
                        def pass_input(**kwargs):
                            return kwargs[param_name]
                        
                        # Create explicit signature with one keyword-only parameter
                        pass_input.__signature__ = Signature([
                            Parameter(param_name, Parameter.KEYWORD_ONLY, annotation=dtype)
                        ])
                        pass_input.__annotations__ = {param_name: dtype, 'return': dtype}
                        return pass_input
                    
                    new_node = NodeObject(
                        name=new_name,
                        action_function=make_pass_function(node.input_dtype, node.name),
                    )
                    self.add_node_to_graph(new_node)
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
            self.add_edge(source, new_target, argument_name=target.name)

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

    def get_node_predecessors(self, node) -> set:
        """Returns all predecessor nodes for the given node.
        
        Args:
            node: The node whose predecessors to retrieve
            
        Returns:
            Set of predecessor nodes
        """
        return set(self.predecessors(node))

    def get_node_successors(self, node) -> set:
        """Returns all successor nodes for the given node.
        
        Args:
            node: The node whose successors to retrieve
            
        Returns:
            Set of successor nodes
        """
        return set(self.successors(node))

    def get_node_predecessor_outputs(self, node) -> dict:
        """Get outputs from immediate predecessor nodes of the given node.
        
        Args:
            node: The node whose predecessor outputs to retrieve
            
        Returns:
            Dictionary mapping argument names (or node names) to predecessor outputs
        """
        predecessors = self.get_node_predecessors(node)
        predecessor_outputs = {}
        for predecessor in predecessors:
            edge = self.edges[predecessor, node]
            argument_name = edge.get('argument_name')
            if argument_name:
                predecessor_outputs[argument_name] = predecessor.output
            else:
                predecessor_outputs[predecessor.name] = predecessor.output
        return predecessor_outputs

    def node_has_skipped_predecessors(self, node) -> bool:
        """Check if any predecessor nodes of the given node have been skipped.
        
        Args:
            node: The node to check
            
        Returns:
            True if any predecessor has been skipped, False otherwise
        """
        predecessors = self.get_node_predecessors(node)
        return any(predecessor.is_skipped() for predecessor in predecessors)

    def is_node_ready(self, node) -> bool:
        """Check if a node is ready to be executed.
        
        A node is ready when:
        1. It is in PENDING state
        2. All its predecessors are either COMPLETED or SKIPPED
        
        Args:
            node: The node to check
            
        Returns:
            True if the node is ready to be executed, False otherwise
        """
        if node.state != NodeState.PENDING:
            return False
        
        predecessors = self.get_node_predecessors(node)
        if not predecessors:
            return True
        
        return all(predecessor.is_completed() or predecessor.is_skipped() 
                   for predecessor in predecessors)

    def validate_node_graph_consistency(self):
        """Validate that all nodes in the graph have this graph as their graph reference."""
        inconsistent_nodes = []
        for node in self.nodes():
            if hasattr(node, 'graph'):
                if node.graph != self:
                    inconsistent_nodes.append(node.name)
        
        if inconsistent_nodes:
            raise ValueError(
                f"Graph inconsistency detected: the following nodes do not reference this graph: "
                f"{inconsistent_nodes}"
            )
        return True

    def can_wire_nodes(self, source: 'BaseNode', target: 'BaseNode') -> bool:
        """Tests if one node can be wired to another node.
        
        Args:
            source: The source node that will output data
            target: The target node that will receive data
            
        Returns:
            bool: True if nodes can be wired, False otherwise
            
        Raises:
            ValueError: If target is not an InputNode
            TypeError: If there is a type mismatch between nodes
        """
        from .nodes import InputNode
        import inspect
        from typing import Any
        import warnings
        
        if not isinstance(target, InputNode):
            raise ValueError(f"Target node must be an input node, got {type(target)}")

        # Get return type from action function signature
        return_annotation = inspect.signature(source.action_function).return_annotation

        # Get expected input type from InputNode
        input_type = target.input_dtype

        # Warn if types cannot be validated
        if return_annotation == inspect.Parameter.empty:
            warnings.warn(
                f"Cannot validate connection: {source.name} -> {target.name}. "
                f"Node {source.name}'s action function lacks return type annotation."
            )
            return True
        elif input_type == Any:
            warnings.warn(
                f"Cannot validate connection: {source.name} -> {target.name}. "
                f"Input node {target.name} accepts Any type."
            )
            return True
        # Validate types if both are specified
        elif return_annotation != inspect.Parameter.empty and input_type != Any:
            if not issubclass(return_annotation, input_type):
                raise TypeError(
                    f"Type mismatch in connection {source.name} -> {target.name}: "
                    f"Node outputs {return_annotation.__name__}, but input node expects {input_type.__name__}"
                )
        return True