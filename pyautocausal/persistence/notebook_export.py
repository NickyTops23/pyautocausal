from typing import Dict, List, Any, Optional
import networkx as nx
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import inspect
from ..orchestration.nodes import Node, InputNode
from ..orchestration.graph import ExecutableGraph

class NotebookExporter:
    """
    Exports an executed graph to a Jupyter notebook format.
    
    This class handles the conversion of a DAG execution into a linear
    sequence of notebook cells that can be executed sequentially.
    """
    
    def __init__(self, graph: ExecutableGraph):
        self.graph = graph
        self.nb = new_notebook()
        self._var_names: Dict[str, str] = {}  # Maps node names to variable names
        
    def _get_topological_order(self) -> List[Node]:
        """Get a valid sequential order of nodes for the notebook."""
        return list(nx.topological_sort(self.graph))
    
    def _create_header(self) -> None:
        """Create header cell with metadata about the graph execution."""
        header = "# Causal Analysis Pipeline\n\n"
        header += "This notebook was automatically generated from a PyAutoCausal pipeline execution.\n\n"
        header += "## Graph Structure\n"
        # Add basic graph info
        header += f"- Number of nodes: {len(self.graph.nodes)}\n"
        
        self.nb.cells.append(new_markdown_cell(header))
    
    def _create_imports_cell(self) -> None:
        """Create cell with all necessary imports."""
        imports = "import pandas as pd\nimport numpy as np\n"
        # TODO: Collect imports from node functions
        self.nb.cells.append(new_code_cell(imports))
    
    def _format_function_definition(self, node: Node) -> str:
        """Format the function definition for a node."""
        if isinstance(node, InputNode):
            return ""
            
        func = node.action_function
        source = inspect.getsource(func)
        
        # If it's a lambda, we need to convert it to a named function
        if source.strip().startswith('lambda'):
            # Get the lambda body
            lambda_body = source.split(':')[1].strip()
            # Create a proper function definition
            source = f"def {node.name}_func(*args, **kwargs):\n    return {lambda_body}"
        
        
        return source
    
    def _get_function_name_from_string(self, function_string: str) -> str:
        """Get the function name from a string."""
        return function_string.split('def')[1].split('(')[0].strip()
    
    def _format_function_execution(self, node: Node, function_string: str) -> str:
        """Format the function execution statement."""
        if isinstance(node, InputNode):
            return f"{node.name}_output = input_data['{node.name}']"
            
        arguments = dict()
        # Get predecessor outputs dictionary
        if not hasattr(node, 'predecessor_outputs'):
            predecessors = node.get_predecessors()
            for predecessor in predecessors:
                edge = self.graph.edges[predecessor, node]
                argument_name = edge.get('argument_name')
                if argument_name:
                    arguments[argument_name] = f"{predecessor.name}_output"
                else:
                    arguments[predecessor.name] = f"{predecessor.name}_output"
        
        
        # Format argument string
        args_str = ", ".join(f"{k}={v}" for k, v in arguments.items()) if arguments else ""

        function_name = self._get_function_name_from_string(function_string)
        
        return f"{node.name}_output = {function_name}({args_str})"
    
    def _create_node_cells(self, node: Node) -> None:
        """Create cells for a single node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
        if hasattr(node, 'condition') and node.condition:
            info += f"\nCondition: {node.condition.description}\n"
        self.nb.cells.append(new_markdown_cell(info))
        
        # Add function definition if not an input node
        if not isinstance(node, InputNode):
            func_def = self._format_function_definition(node)
            self.nb.cells.append(new_code_cell(func_def))
        
            # Add execution cell
            exec_code = self._format_function_execution(node, func_def)
            self.nb.cells.append(new_code_cell(exec_code))
    
    def export_notebook(self, filepath: str) -> None:
        """
        Export the graph execution as a Jupyter notebook.
        
        Args:
            filepath: Path where the notebook should be saved
        """
        # Create header
        self._create_header()
        
        # Create imports
        self._create_imports_cell()
        
        # Process nodes in topological order
        for node in self._get_topological_order():
            if node.is_completed() and ~isinstance(node, InputNode):  # Only include executed nodes
                self._create_node_cells(node)
        
        # Save the notebook
        with open(filepath, 'w') as f:
            nbformat.write(self.nb, f) 