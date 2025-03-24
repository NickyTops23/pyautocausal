from typing import Dict, List, Any, Optional, Tuple
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
    
    def _is_exposed_wrapper(self, func) -> bool:
        """Check if a function is decorated with expose_in_notebook."""
        return hasattr(func, '_notebook_export_info') and func._notebook_export_info.get('is_wrapper', False)
    
    def _get_exposed_target_info(self, func) -> Tuple[Any, Dict[str, str]]:
        """Get the target function and argument mapping from an exposed wrapper."""
        if not self._is_exposed_wrapper(func):
            return None, {}
        
        info = func._notebook_export_info
        return info.get('target_function'), info.get('arg_mapping', {})
    
    def _format_function_definition(self, node: Node) -> str:
        """Format the function definition for a node."""
        if isinstance(node, InputNode):
            return ""
            
        func = node.action_function
        
        # Check if this is an exposed wrapper function
        if self._is_exposed_wrapper(func):
            target_func, arg_mapping = self._get_exposed_target_info(func)
            
            # Get the source of both functions
            wrapper_source = inspect.getsource(func)
            target_source = inspect.getsource(target_func)
            
            # Format a comment explaining the wrapper relationship
            mapping_str = ", ".join([f"'{wrapper}' → '{target}'" for wrapper, target in arg_mapping.items()])
            comment = f"# This node uses a wrapper function that calls a target function with adapted arguments\n"
            comment += f"# Argument mapping: {mapping_str}\n\n"
            
            # Get the target function's name
            target_name = target_func.__name__
            
            # Create imports if the target function is from an external module
            module_name = target_func.__module__
            if module_name != '__main__':
                import_statement = f"from {module_name} import {target_name}\n\n"
                comment += import_statement
            
            # Add both the target and wrapper
            return comment + target_source + "\n\n" + wrapper_source
        
        # Handle lambdas
        source = inspect.getsource(func)
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
        
        func = node.action_function
        is_wrapper = self._is_exposed_wrapper(func)
        
        arguments = dict()
        # Get predecessor outputs dictionary
        predecessors = self.graph.get_node_predecessors(node)
        if predecessors:
            for predecessor in predecessors:
                edge = self.graph.edges[predecessor, node]
                argument_name = edge.get('argument_name')
                if argument_name:
                    arguments[argument_name] = f"{predecessor.name}_output"
                else:
                    arguments[predecessor.name] = f"{predecessor.name}_output"
        
        # Format argument string
        args_str = ", ".join(f"{k}={v}" for k, v in arguments.items()) if arguments else ""
        
        # For wrapped functions, add a comment showing how to call the target directly
        if is_wrapper:
            _, arg_mapping = self._get_exposed_target_info(func)
            
            # For target function call, map the arguments according to the mapping
            target_args = {}
            for wrapper_arg, wrapper_value in arguments.items():
                target_arg = arg_mapping.get(wrapper_arg, wrapper_arg)
                target_args[target_arg] = wrapper_value
            
            target_args_str = ", ".join(f"{k}={v}" for k, v in target_args.items()) if target_args else ""
            
            # Get the target function's name
            target_func = func._notebook_export_info['target_function']
            target_name = target_func.__name__
            
            # Create both function calls, with the direct call commented out
            function_name = self._get_function_name_from_string(function_string)
            wrapper_call = f"{node.name}_output = {function_name}({args_str})"
            target_call = f"# Alternatively, call the target function directly:\n# {node.name}_output = {target_name}({target_args_str})"
            
            return wrapper_call + "\n" + target_call
        
        # Normal case - just call the function
        function_name = self._get_function_name_from_string(function_string)
        return f"{node.name}_output = {function_name}({args_str})"
    
    def _create_node_cells(self, node: Node) -> None:
        """Create cells for a single node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
        if node.node_description:
            info += f"{node.node_description}\n"
        self.nb.cells.append(new_markdown_cell(info))
        
        # Add function definition if not an input node
        if not isinstance(node, InputNode):
            func_def = self._format_function_definition(node)
            self.nb.cells.append(new_code_cell(func_def))
        
            # Add execution cell
            exec_code = self._format_function_execution(node, func_def)
            self.nb.cells.append(new_code_cell(exec_code))
    
    def _create_input_node_cells(self, node: InputNode) -> None:
        """Create cells for a single input node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
        if node.node_description:
            info += f"{node.node_description}\n"
        self.nb.cells.append(new_markdown_cell(info))
        
        # Add execution cell which is just a comment telling the user to provide the input
        exec_code = f"# TODO: Load your input data for '{node.name}' here\n{node.name}_output = None  # Replace with your data"
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
            if node.is_completed():
                if isinstance(node, InputNode):
                    self._create_input_node_cells(node)
                else:
                    self._create_node_cells(node)
        
        # Save the notebook
        with open(filepath, 'w') as f:
            nbformat.write(self.nb, f) 