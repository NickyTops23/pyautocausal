import pandas as pd
import statsmodels.api as sm
import io
from typing import Callable, Optional
from abc import abstractmethod
from ..orchestration.nodes import Node, ExecutableGraph, OutputConfig
from ..persistence.output_config import OutputType

class LibraryNode(Node):
    """Base class for standardized nodes with configurable output and conditions."""
    
    def __init__(self, 
                 graph: ExecutableGraph, 
                 name: str, 
                 condition: Optional[Callable[[pd.DataFrame], bool]] = None, 
                 skip_reason: str = "", 
                 save_output: bool = True, 
                 output_filename: str = "", 
                 output_type: OutputType = OutputType.TEXT):
        
        if condition is None:
            condition = self.condition
            
        super().__init__(
            name=name,
            graph=graph,
            action_function=self.action,
            condition=condition,
            skip_reason=skip_reason,
            output_config=OutputConfig(
                save_output=save_output,
                output_filename=output_filename,
                output_type=output_type
            )
        )
    
    @abstractmethod
    def action(self, df: pd.DataFrame) -> str:
        """Define the standard action for this node."""
        pass
    
    @abstractmethod
    def condition(self, df: pd.DataFrame) -> bool:
        """Define the standard condition for this node."""
        pass


class OLSNode(LibraryNode):
    """Node for Ordinary Least Squares regression analysis."""
    
    def __init__(self, 
                 graph: ExecutableGraph, 
                 name: str = "OLS Treatment Effect", 
                 condition: Optional[Callable[[pd.DataFrame], bool]] = None,
                 skip_reason: str = "Sample size too large for OLS",
                 save_output: bool = True,
                 output_filename: str = "ols_treatment_effect"):
        super().__init__(
            graph=graph,
            name=name,
            condition=condition,
            skip_reason=skip_reason,
            save_output=save_output,
            output_filename=output_filename
        )
   
    @staticmethod
    def action(df: pd.DataFrame) -> str:
        """Estimate treatment effect using OLS regression."""
        required_columns = ['y', 'treat']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        if not pd.api.types.is_numeric_dtype(df['y']) or not pd.api.types.is_numeric_dtype(df['treat']):
            raise TypeError("'y' and 'treat' must be numeric types.")
        
        y = df['y']
        X = pd.concat([df['treat'], df.drop(columns=['y', 'treat'])], axis=1)
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()
        
        buffer = io.StringIO()
        buffer.write(str(results.summary()))
        return buffer.getvalue()
        
    @staticmethod
    def condition(df: pd.DataFrame) -> bool:
        """Check if sample size is appropriate for OLS."""
        return len(df) <= 100


class DoubleMLNode(LibraryNode):
    """Node for Double Machine Learning analysis."""
    
    def __init__(self, 
                 graph: ExecutableGraph, 
                 name: str = "DoublesML Treatment Effect", 
                 condition: Optional[Callable[[pd.DataFrame], bool]] = None,
                 skip_reason: str = "Sample size too small for Double ML", 
                 save_output: bool = True,
                 output_filename: str = "doubleML_treatment_effect"):
        super().__init__(
            graph=graph,
            name=name,
            condition=condition,
            skip_reason=skip_reason,
            save_output=save_output,
            output_filename=output_filename
        )

    @staticmethod
    def action(df: pd.DataFrame) -> str:
        """Estimate treatment effect using Double Machine Learning."""
        required_columns = ['y', 'treat']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        if not pd.api.types.is_numeric_dtype(df['y']) or not pd.api.types.is_numeric_dtype(df['treat']):
            raise TypeError("'y' and 'treat' must be numeric types.")
        
        y = df['y']
        t = df['treat']
        X = df.drop(columns=['y', 'treat'])
        
        model_t = sm.OLS(t, sm.add_constant(X)).fit()
        model_y = sm.OLS(y, sm.add_constant(X)).fit()
        
        t_residual = t - model_t.predict(sm.add_constant(X))
        y_residual = y - model_y.predict(sm.add_constant(X))
        
        effect_model = sm.OLS(
            y_residual,
            sm.add_constant(pd.Series(t_residual, name='treat'))
        ).fit()

        buffer = io.StringIO()
        buffer.write(str(effect_model.summary()))
        return buffer.getvalue()

    @staticmethod
    def condition(df: pd.DataFrame) -> bool:
        """Check if sample size is appropriate for Double ML."""
        return len(df) > 100


class PassthroughNode(LibraryNode):
    """Node for transforming data."""
    def __init__(self, graph: ExecutableGraph, name: str = "Passthrough Data", condition: Optional[Callable[[pd.DataFrame], bool]] = None, skip_reason: str = "Sample size too small for Transform Data", save_output: bool = True, output_filename: str = "transform_data"):
        super().__init__(graph=graph, name=name, condition=condition, skip_reason=skip_reason, save_output=save_output, output_filename=output_filename)

    def action(self, df: pd.DataFrame) -> str:
        return df

    def condition(self, df: pd.DataFrame) -> bool:
        return True
