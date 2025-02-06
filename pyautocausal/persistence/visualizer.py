import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_graph(graph, save_path, return_positions=False, return_labels=False):
    """
    Visualize the provided graph as a static image in a bottom-to-top (decision tree) layout.
    
    Parameters:
        graph (nx.DiGraph): The directed graph (e.g., an instance of ExecutableGraph) to visualize.
        save_path (str): The file path where the resulting image will be saved.
        return_positions (bool): If True, return the node positions (for testing).
        return_labels (bool): If True, return the node labels (for testing).
        
    Returns:
        dict: Node positions if return_positions is True
        dict: Node labels if return_labels is True
    """
    try:
        # Validate input
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Input must be a NetworkX DiGraph")
        
        if len(graph) == 0:
            raise ValueError("Graph is empty")
            
        # Close all existing figures
        plt.close('all')
        
        # Get input nodes (nodes with no predecessors)
        input_nodes = [n for n in graph.nodes() if len(list(graph.predecessors(n))) == 0]
        if not input_nodes:
            raise ValueError("Graph must have at least one input node")
        
        # Group nodes by their level
        levels = defaultdict(list)
        for node in graph.nodes():
            paths = []
            for input_node in input_nodes:
                try:
                    node_paths = list(nx.all_simple_paths(graph, input_node, node))
                    paths.extend([len(p) for p in node_paths])
                except nx.NetworkXNoPath:
                    continue
            
            level = max(paths) if paths else 0
            levels[level].append(node)
        
        if not levels:
            raise ValueError("Failed to calculate node levels")
        
        # Calculate positions
        pos = {}
        max_level = max(levels.keys()) if levels else 0
        
        for level, nodes in levels.items():
            width = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - (width - 1) / 2) / max(width - 1, 1)
                y = -level / max(max_level, 1)
                pos[node] = (x, y)
        
        # Create labels dictionary using just the node names
        labels = {node: node.name for node in graph.nodes()}
        
        # Create the figure and draw
        plt.figure(figsize=(12, 8))
        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            arrows=True,
            node_size=2000,
            node_color='lightblue',
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            arrowsize=20
        )
        plt.title("Executable Graph Visualization")
        
        # Save the static image
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        # Return values for testing if requested
        if return_positions and return_labels:
            return pos, labels
        elif return_positions:
            return pos
        elif return_labels:
            return labels
        
    except Exception as e:
        logger.error(f"Error visualizing graph: {str(e)}")
        raise

# For convenience, if someone runs this module directly

   