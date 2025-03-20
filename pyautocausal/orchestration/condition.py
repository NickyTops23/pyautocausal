from typing import Callable, Any
import types
class Condition:
    """Class representing an execution condition with description."""
    
    def __init__(self, condition_func: Callable[..., bool], description: str):
        """
        Initialize a condition.
        
        Args:
            condition_func: Function that evaluates to True/False
              can be a lambda function only if it takes a single argument
              otherwise it should take **kwargs
            description: Human-readable description of the condition
        """
        
        # check if condition_func is a lambda function
        if isinstance(condition_func, types.LambdaType):
            # check if it takes a single argument
            if condition_func.__code__.co_argcount != 1:
                raise ValueError("Lambda function must take a single argument")

        self.condition_func = condition_func
        self.description = description
    
    def evaluate(self, **kwargs) -> bool:
        """
        Evaluate the condition with given arguments. If the condition function is a lambda function then pass the single argument.
        
        Args:
            **kwargs: Arguments to pass to condition function
            
        Returns:
            bool: Result of condition evaluation
        """
        if isinstance(self.condition_func, types.LambdaType):
            # validate that there is max one argument
            if len(self.condition_func.__code__.co_varnames) > 1:
                raise ValueError("Lambda function must take a single argument")
            elif len(self.condition_func.__code__.co_varnames) == 0:
                return self.condition_func()
            else:
                return self.condition_func(kwargs[self.condition_func.__code__.co_varnames[0]])
        else:
            return self.condition_func(**kwargs)
    

def create_condition(condition_func: Callable[..., bool], description: str) -> Condition:
    """Create a condition object"""
    return Condition(condition_func, description)