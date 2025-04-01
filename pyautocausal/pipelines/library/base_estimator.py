import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """
    Base class for all estimators in the pyautocausal package.
    
    This abstract base class defines the common interface that all estimators
    should implement, including methods for model fitting (action) and
    result formatting (output).
    """
    
    @classmethod
    @abstractmethod
    def action(cls, 
               df: pd.DataFrame, 
               specification: Optional[Dict] = None, 
               weights: Optional[np.ndarray] = None, 
               **kwargs) -> Any:
        """
        Fit the model and estimate treatment effects.
        
        Args:
            df: DataFrame with the data for estimation
            specification: Dictionary with model specification details
            weights: Optional sample weights to apply
            **kwargs: Additional estimator-specific parameters
                
        Returns:
            Fitted model object
        """
        pass
    
    @classmethod
    @abstractmethod
    def output(cls, model: Any) -> str:
        """
        Format the model results into a readable string.
        
        Args:
            model: Fitted model object from the action method
                
        Returns:
            Formatted string with model results
        """
        pass
    
    @classmethod
    def compute_weights(cls, df: pd.DataFrame, **kwargs) -> Union[np.ndarray, tuple]:
        """
        Optional method to compute weights that would be used in the action method.
        
        This default implementation returns None, indicating no weights.
        Subclasses can override this to implement weighting methods.
        
        Args:
            df: DataFrame with the data for weight computation
            **kwargs: Additional weight computation parameters
                
        Returns:
            Weights array or tuple with weights and additional information
        """
        return None 