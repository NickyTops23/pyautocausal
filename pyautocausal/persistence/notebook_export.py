from typing import Dict, List, Any, Optional, Tuple, Callable
import networkx as nx
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import inspect
from ..orchestration.nodes import Node, InputNode, DecisionNode
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
        # get module this is run from
        self.this_module = inspect.getmodule(inspect.getouterframes(inspect.currentframe())[1][0])
        
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
        
        if hasattr(node, 'notebook_function') and node.notebook_function is not None and node.is_completed():
            return ""

        if isinstance(node, DecisionNode):
            func = node.condition
        else:
            func = node.action_function
        
        # Check if this is an exposed wrapper function
        if self._is_exposed_wrapper(func):
            target_func, arg_mapping = self._get_exposed_target_info(func)
            
            target_source = inspect.getsource(target_func)
            
            # Remove the @make_transformable decorator lines
            target_lines = target_source.split('\n')
            filtered_lines = [line for line in target_lines if not line.strip().startswith('@make_transformable')]
            target_source = '\n'.join(filtered_lines)

            # Format a comment explaining the wrapper relationship
            mapping_str = ", ".join([f"'{wrapper}' → '{target}'" for wrapper, target in arg_mapping.items()])
            comment = f"# This node uses a wrapper function that calls a target function with adapted arguments\n"
            comment += f"# Argument mapping: {mapping_str}\n\n"
            
            # Get the target function's name
            target_name = target_func.__name__
            
            # Create imports if the target function is from an external module
            # TODO: This doesn't work for functions defined in pyautocausal
            module_name = target_func.__module__
            if module_name != self.this_module.__dict__['__name__']:
                import_statement = f"from {module_name} import {target_name}\n\n"
                target_source = import_statement + target_source
            
            # Add both the target and wrapper
            return target_source
        
        # Handle lambdas
        source = inspect.getsource(func)
        if source.strip().startswith('lambda'):
            # Get the lambda body
            lambda_body = source.split(':')[1].strip()
            # Create a proper function definition
            source = f"def {node.name}_func(*args, **kwargs):\n    return {lambda_body}"
        
        # Handle annotations before function definition
        annotations = []
        non_annotation_lines = []
        func_lines = source.split('\n')
        for line in func_lines:
            if line.strip().startswith('@'):
                annotations.append(line.strip())
            else:
                non_annotation_lines.append(line.strip())
        
        return '\n'.join(annotations) + '\n' + '\n'.join(non_annotation_lines)
    
    def _get_function_name_from_string(self, function_string: str) -> str:
        """Get the function name from a string."""
        return function_string.split('def')[1].split('(')[0].strip()
    
    
    def _find_argument_source_nodes(self, current_node: Node) -> Dict[str, str]:
        """
        Traces backwards from a node to find the non-decision-node ancestors
        that provide data. Effectively finds the origins of data flowing into
        the current_node, ignoring intermediate DecisionNodes.

        Args:
            current_node: The node to start tracing backwards from.

        Returns:
            A dictionary mapping the names of ancestor source node names to the nodes 
            (e.g., {'df': Node, 'settings': Node}).
            This indicates which data sources are potentially available.
        """
        source_nodes = {}
        visited = set()
        queue = list(self.graph.predecessors(current_node)) # Use list for queue behavior

        while queue:
            predecessor = queue.pop(0) # FIFO

            if predecessor in visited:
                continue
            visited.add(predecessor)

            if isinstance(predecessor, DecisionNode):
                # If it's a decision node, add its predecessors to the queue
                # to continue tracing backwards *through* it.
                for decision_predecessor in self.graph.predecessors(predecessor):
                    if decision_predecessor not in visited:
                        queue.append(decision_predecessor)
            else:
                # If it's a regular node or an input node, it's a source.
                # We store its name as an available data source.
                source_nodes[predecessor.name] = predecessor

        return source_nodes

    def _resolve_function_arguments(self, node: Node, func: Callable) -> Dict[str, str]:
        """Resolve the arguments for a node's function."""

        is_wrapper = self._is_exposed_wrapper(func)
        
        arguments = dict()  
        #TODO: Handle default arguments for non-wrapper functions
        

        # For wrapper functions, use the argument mapping to map the arguments to the predecessor node names
        if is_wrapper:
            _, arg_mapping = self._get_exposed_target_info(func)
            for predecessor_name, func_param in arg_mapping.items():
                arguments[func_param] = f"{predecessor_name}_output"
        else:
            arg_mapping = dict()

        # Get predecessor nodes that provide data, ignoring decision nodes in between
        arg_source_nodes = self._find_argument_source_nodes(node)        

        # Handle the arguments that are not transformed
        for arg_name, _ in arg_source_nodes.items():
            if arg_name not in arg_mapping:
                arguments[arg_name] = f"{arg_name}_output"

        #TODO: Add check that all required arguments are present and all provided arguments are part of arguments
        #TODO: Handle run-context arguments

        return arguments

    def _format_notebook_function(self, node: Node) -> str:
        """Format the notebook function for a node.
        Notebook functions return string representations of the function definition,
        where arguments are denoted by the argument name followed by "[argument_name]_argument"

        This function takes the string representation of the notebook function and
        resolves the arguments to the actual node names.
        """
        
        arguments = self._resolve_function_arguments(node, node.action_function)

        node_result = node.output.result_dict[node.name]
        notebook_display_string = node.notebook_function(node_result)

        # Replace the argument placeholders with the actual node names
        for arg_name, predecessor_output in arguments.items():
            notebook_display_string = notebook_display_string.replace(f"{arg_name}_argument", f"{predecessor_output}")

        return f"{node.name}_output = {notebook_display_string}"
    
    def _format_function_execution(self, node: Node, function_string: str) -> str:
        """Format the function execution statement."""
        if isinstance(node, InputNode):
            return f"{node.name}_output = input_data['{node.name}']"

        # TODO: Handle cases where we want to show condition functions
        func = node.action_function

        arguments = self._resolve_function_arguments(node, func)
        
        repr_string_noop = lambda x: repr(x) if not isinstance(x, str) else x
        
        # Format argument string
        args_str = ", ".join(f"{k}={repr_string_noop(v)}" for k, v in arguments.items()) if arguments else ""
        
        
        is_wrapper = self._is_exposed_wrapper(func)
        
        # For wrapped functions, add a comment showing how to call the target directly
        if is_wrapper:
            _, arg_mapping = self._get_exposed_target_info(func)
            
            # For target function call, map the arguments according to the mapping
            target_args = {}
            for wrapper_arg, wrapper_value in arguments.items():
                target_arg = arg_mapping.get(wrapper_arg, wrapper_arg)
                target_args[target_arg] = wrapper_value
            
            target_args_str = ", ".join(f"{k}={repr(v)}" for k, v in target_args.items()) if target_args else ""
            
            # Get the target function's name
            target_func = func._notebook_export_info['target_function']
            target_name = target_func.__name__
            
            # Create both function calls, with the direct call commented out
            function_name = self._get_function_name_from_string(function_string)
            wrapper_call = f"{node.name}_output = {function_name}({args_str})"
            
            return wrapper_call
        
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
        
        # Add function definition if not an input node and no notebook function is defined
        if hasattr(node, 'notebook_function') and node.notebook_function is not None and node.is_completed():
            # Add execution cell
            notebook_display_code = self._format_notebook_function(node)
            self.nb.cells.append(new_code_cell(notebook_display_code))
    
        elif not isinstance(node, InputNode):
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
            # Skip decision nodes
            if isinstance(node, DecisionNode):
                continue
                
            if node.is_completed():
                if isinstance(node, InputNode):
                    self._create_input_node_cells(node)
                else:
                    self._create_node_cells(node)
        
        # Save the notebook
        with open(filepath, 'w') as f:
            nbformat.write(self.nb, f) 