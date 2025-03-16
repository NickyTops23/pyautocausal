from typing import Dict, List, Any, Optional
import networkx as nx
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import inspect
import logging
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
        """
        Format the function definition for a node.
        
        This method handles different types of functions:
        - Lambda functions
        - Class methods (bound methods)
        - Static methods
        - Regular functions
        
        Returns a string with the function definition that can be included in the notebook.
        """
        if isinstance(node, InputNode):
            return ""
            
        func = node.action_function
        
        # Case 1: Lambda function
        if func.__name__ == '<lambda>':
            self.graph.logger.info(f"Node '{node.name}': Using generic wrapper for lambda function")
            # For lambda functions, create a simple wrapper function with the node name
            return f"def {node.name}_func(df):\n    # Lambda function wrapper\n    return df"
        
        # Case 2: Bound method (instance method of a class)
        if hasattr(func, '__self__') and func.__self__ is not None:
            try:
                source = inspect.getsource(func.__func__)
                self.graph.logger.debug(f"Node '{node.name}': Successfully extracted source from bound method")
            except (TypeError, OSError) as e:
                self.graph.logger.error(f"Node '{node.name}': Could not extract source from bound method: {str(e)}")
                raise ValueError(f"Could not extract source code for bound method in node '{node.name}': {str(e)}")
        # Case 3: Regular function or static method
        else:
            try:
                source = inspect.getsource(func)
                self.graph.logger.debug(f"Node '{node.name}': Successfully extracted source from function")
            except (TypeError, OSError) as e:
                self.graph.logger.error(f"Node '{node.name}': Could not extract source from function: {str(e)}")
                raise ValueError(f"Could not extract source code for function in node '{node.name}': {str(e)}")
        
        # If we got the source but it's a lambda, convert it to a named function
        if source.strip().startswith('lambda'):
            try:
                lambda_body = source.split(':')[1].strip()
                source = f"def {node.name}_func(*args, **kwargs):\n    return {lambda_body}"
                self.graph.logger.debug(f"Node '{node.name}': Converted lambda to named function")
            except IndexError as e:
                self.graph.logger.error(f"Node '{node.name}': Failed to parse lambda function: {str(e)}")
                # If we can't parse the lambda properly, use a placeholder
                raise ValueError(f"Failed to parse lambda function in node '{node.name}': {str(e)}")
        
        return source
    
    def _get_function_name_from_string(self, function_string: str) -> str:
        """
        Get the function name from a string or function object.
        
        Args:
            function_string: Either a string containing function definition or a function object
            
        Returns:
            The extracted function name or a generic name
        """
        # Case 1: String with function definition
        if isinstance(function_string, str) and 'def' in function_string:
            try:
                name = function_string.split('def')[1].split('(')[0].strip()
                self.graph.logger.debug(f"Extracted function name '{name}' from definition")
                return name
            except IndexError:
                self.graph.logger.warning("Failed to extract function name from definition")
                return "function"  # Fallback if parsing fails
        
        # Case 2: Function object with __name__ attribute
        elif hasattr(function_string, '__name__'):
            name = function_string.__name__
            self.graph.logger.debug(f"Using function name '{name}' from __name__ attribute")
            return name
        
        # Case 3: Default case - use generic name
        else:
            self.graph.logger.warning("Could not determine function name, using generic 'function'")
            return "function"
    
    def _format_function_execution(self, node: Node, function_string: str) -> str:
        """Format the function execution statement."""
        if isinstance(node, InputNode):
            return f"{node.name}_output = input_data['{node.name}']"
            
        # Get predecessor outputs for function arguments
        arguments = self._get_predecessor_arguments(node)
        
        # Format argument string
        args_str = ", ".join(f"{k}={v}" for k, v in arguments.items()) if arguments else ""

        # Get appropriate function name based on the context
        if isinstance(function_string, str) and 'def' in function_string:
            # If we have a function definition string, extract the name
            function_name = self._get_function_name_from_string(function_string)
        else:
            # Otherwise use the node name with _func suffix
            function_name = f"{node.name}_func"
            self.graph.logger.debug(f"Using '{function_name}' for node '{node.name}'")
        
        return f"{node.name}_output = {function_name}({args_str})"
    
    def _get_predecessor_arguments(self, node: Node) -> dict:
        """Get the arguments from predecessor nodes."""
        arguments = {}
        predecessors = node.get_predecessors()
        
        if predecessors:
            for predecessor in predecessors:
                edge = self.graph.edges[predecessor, node]
                argument_name = edge.get('argument_name')
                if argument_name:
                    arguments[argument_name] = f"{predecessor.name}_output"
                else:
                    arguments[predecessor.name] = f"{predecessor.name}_output"
                    
        return arguments
    
    def _create_node_cells(self, node: Node) -> None:
        """Create cells for a single node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
        if hasattr(node, 'condition') and node.condition:
            info += f"\nCondition: {node.condition.description}\n"
        self.nb.cells.append(new_markdown_cell(info))
        
        # Add function definition if not an input node
        if not isinstance(node, InputNode):
            # No more try-except - errors will propagate up
            self.graph.logger.info(f"Processing node '{node.name}' for notebook export")
            func_def = self._format_function_definition(node)
            self.nb.cells.append(new_code_cell(func_def))
            
            # Add execution cell
            exec_code = self._format_function_execution(node, func_def)
            self.nb.cells.append(new_code_cell(exec_code))
    
    def _create_input_node_cells(self, node: InputNode) -> None:
        """Create cells for a single input node's execution."""
        # Add markdown cell with node info
        info = f"## Node: {node.name}\n"
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
        self.graph.logger.info(f"Starting notebook export to {filepath}")
        
        # Create header
        self._create_header()
        
        # Create imports
        self._create_imports_cell()
        
        # Process nodes in topological order
        nodes = self._get_topological_order()
        self.graph.logger.info(f"Processing {len(nodes)} nodes in topological order")
        
        for node in nodes:
            if node.is_completed():
                if isinstance(node, InputNode):
                    self._create_input_node_cells(node)
                else:
                    self._create_node_cells(node)
        
        # Save the notebook
        with open(filepath, 'w') as f:
            nbformat.write(self.nb, f)

            
        self.graph.logger.info(f"Notebook successfully exported to {filepath}") 