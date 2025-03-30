import pandas as pd
import statsmodels.api as sm
import io
from typing import Callable, Optional
from abc import abstractmethod
from pyautocausal.orchestration.nodes import Node, ExecutableGraph, OutputConfig
from pyautocausal.persistence.output_config import OutputType
from pyautocausal.orchestration.condition import Condition

class LibraryNode(Node):
    """Base class for standardized nodes with configurable output and conditions."""
    
    def __init__(self, 
                 name: str = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = True,
                 output_filename: str = "", 
                 output_type: OutputType = OutputType.TEXT,
                 graph: Optional[ExecutableGraph] = None):
        
        if condition is None:
            condition = self.condition()
            
        super().__init__(
            name=name or self.__class__.__name__,
            graph=graph,
            action_function=self.action,
            condition=condition,
            save_node=save_node,
            output_config=OutputConfig(
                output_filename=output_filename,
                output_type=output_type
            ) if save_node else None
        )
    
    @abstractmethod
    def action(self, df: pd.DataFrame) -> str:
        """Define the standard action for this node."""
        pass
    
    @abstractmethod
    def condition(self) -> Condition:
        """Define the default condition for this node."""
        pass




