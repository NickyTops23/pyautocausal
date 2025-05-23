from typing import Dict, List, Any, Optional, Tuple, Callable
import networkx as nx
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import inspect
import ast
from ..orchestration.nodes import Node, InputNode, DecisionNode
from ..orchestration.graph import ExecutableGraph
from .visualizer import visualize_graph
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
        self.needed_imports = set()
    def _get_topological_order(self) -> List[Node]:
        """Get a valid sequential order of nodes for the notebook."""
        return list(nx.topological_sort(self.graph))
    
    def _create_header(self) -> None:
        """Create header cell with metadata about the graph execution."""
        header = "# Causal Analysis Pipeline\n\n"
        header += "This notebook was automatically generated from a PyAutoCausal pipeline execution.\n\n"
        
        self.nb.cells.append(new_markdown_cell(header))
    
    def _create_imports_cell(self, cell_index: int) -> None:
        """Create cell with all necessary imports."""
        all_imports = "\n".join(self.needed_imports)
        # insert cell at cell_index
        self.nb.cells.insert(cell_index, new_code_cell(all_imports))
    
    def _is_exposed_wrapper(self, func) -> bool:
        """Check if a function is decorated with expose_in_notebook."""
        return hasattr(func, '_notebook_export_info') and func._notebook_export_info.get('is_wrapper', False)
    
    def _get_exposed_target_info(self, func) -> Tuple[Any, Dict[str, str]]:
        """Get the target function and argument mapping from an exposed wrapper."""
        if not self._is_exposed_wrapper(func):
            return None, {}
        
        info = func._notebook_export_info
        return info.get('target_function'), info.get('arg_mapping', {})
    
    def _get_function_imports(self, func) -> None:
        """Extract import statements needed for a given function."""
        try:
            # First analyze the module to track imports and their aliases
            module = inspect.getmodule(func)
            try:
                module_source = inspect.getsource(module)
                module_tree = ast.parse(module_source)
                
                # Track imports and their aliases
                import_aliases = {}  # Maps aliases to (module, original_name)
                
                for node in ast.walk(module_tree):
                    # Handle from x import y as z
                    if isinstance(node, ast.ImportFrom):
                        module_name = node.module
                        for name in node.names:
                            alias = name.asname or name.name
                            import_aliases[alias] = (module_name, name.name)
                    
                    # Handle import x as y
                    elif isinstance(node, ast.Import):
                        for name in node.names:
                            alias = name.asname or name.name
                            import_aliases[alias] = (name.name, None)
            except Exception:
                # If module source parsing fails, continue with function analysis only
                import_aliases = {}
            
            # Get function source code and parse it
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            # Process Name nodes (simple references)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    name = node.id
                    
                    # Check if it's an imported alias
                    if name in import_aliases:
                        module_name, original_name = import_aliases[name]
                        if original_name:
                            self.needed_imports.add(f"from {module_name} import {original_name}{' as ' + name if original_name != name else ''}")
                        else:
                            self.needed_imports.add(f"import {module_name}{' as ' + name if module_name != name else ''}")
                    
                    # Check if it's a function/object from another module
                    elif name in module.__dict__:
                        obj = module.__dict__[name]
                        if hasattr(obj, '__module__'):
                            module_name = obj.__module__
                            if (module_name != 'builtins' and 
                                not module_name.startswith('_')):
                                self.needed_imports.add(f"from {module_name} import {name}")
                
                # Handle attribute access (like package.function)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    pkg_name = node.value.id
                    attr_name = node.attr
                    
                    # Check if the base is an imported package
                    if pkg_name in import_aliases:
                        module_name, _ = import_aliases[pkg_name]
                        self.needed_imports.add(f"import {module_name}{' as ' + pkg_name if module_name != pkg_name else ''}")
                    
                    # Check if it's a direct module in the module namespace
                    elif pkg_name in module.__dict__:
                        obj = module.__dict__[pkg_name]
                        if hasattr(obj, '__name__'):
                            module_name = obj.__name__
                            self.needed_imports.add(f"import {module_name}{' as ' + pkg_name if module_name != pkg_name else ''}")
    
        
        except Exception as e:
            # Fallback in case of any errors
            return f"# Import analysis failed: {str(e)}\n"
    
    def _format_function_definition(self, node: Node) -> str:
        """Format the function definition for a node."""
        if isinstance(node, InputNode):
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
            
            # Create imports using AST analysis
            self._get_function_imports(target_func)
            
            # Add both the target and wrapper with proper imports
            return comment + target_source
        
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
    
        if not isinstance(node, InputNode):
            func_def = self._format_function_definition(node)
            self.nb.cells.append(new_code_cell(func_def))
        
            exec_code = self._format_function_execution(node, func_def)
            self.nb.cells.append(new_code_cell(exec_code))
        
            if node.display_function:
                notebook_display_code = self._format_display_function_call(node)
                self.nb.cells.append(new_code_cell(notebook_display_code))
    
    def _format_display_function_call(self, node: Node) -> str:
        """Format the display function call for a node. This simply
        calls the display function on the node's output."""
        return f"# Display Result\n{node.display_function.__name__}({node.name}_output)"

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
        
        # Get graph visualization without title
        markdown_content = visualize_graph(self.graph)
        # Remove the title line and empty line after it
        markdown_content = "\n".join(markdown_content.split("\n")[1:])
        self.nb.cells.append(new_markdown_cell(markdown_content))

        end_of_markdown = len(self.nb.cells)

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

        self._create_imports_cell(end_of_markdown)
        
        # Save the notebook
        with open(filepath, 'w') as f:
            nbformat.write(self.nb, f) 