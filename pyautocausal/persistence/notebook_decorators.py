from functools import wraps
from typing import Callable, Dict, Any, Optional

def expose_in_notebook(target_function: Callable, arg_mapping: Optional[Dict[str, str]] = None):
    """
    Decorator that marks a function as a thin wrapper and specifies the target function
    that should be exposed in notebook exports.
    
    This decorator helps the notebook exporter understand when a function is just a wrapper
    that adapts argument names between nodes and the actual implementation.
    
    Args:
        target_function: The actual function to expose in notebook exports
        arg_mapping: Dictionary mapping wrapper function parameters to target function parameters
                     e.g., {'data': 'df'} means wrapper param 'data' maps to target param 'df'
    
    Returns:
        The decorated function, with metadata attached for the notebook exporter
    
    Example:
        @expose_in_notebook(target_function=complex_stats_function, arg_mapping={'data': 'df'})
        def stats_action(data):
            return complex_stats_function(df=data)
    """
    def decorator(wrapper_func):
        # Attach metadata to the wrapper function
        wrapper_func._notebook_export_info = {
            'is_wrapper': True,
            'target_function': target_function,
            'arg_mapping': arg_mapping or {}
        }
        
        @wraps(wrapper_func)
        def wrapped_function(*args, **kwargs):
            return wrapper_func(*args, **kwargs)
        
        # Transfer metadata to the wrapped function
        wrapped_function._notebook_export_info = wrapper_func._notebook_export_info
        return wrapped_function
    
    return decorator 