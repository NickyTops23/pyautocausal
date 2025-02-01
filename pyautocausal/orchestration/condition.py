from typing import Callable, Any

class Condition:
    """Class representing an execution condition with description."""
    
    def __init__(self, condition_func: Callable[..., bool], description: str):
        """
        Initialize a condition.
        
        Args:
            condition_func: Function that evaluates to True/False
            description: Human-readable description of the condition
        """
        self.condition_func = condition_func
        self.description = description
    
    def evaluate(self, **kwargs) -> bool:
        """
        Evaluate the condition with given arguments.
        
        Args:
            **kwargs: Arguments to pass to condition function
            
        Returns:
            bool: Result of condition evaluation
        """
        return self.condition_func(**kwargs) 
    

def create_condition(condition_func: Callable[..., bool], description: str) -> Condition:
    """Create a condition object"""
    return Condition(condition_func, description)