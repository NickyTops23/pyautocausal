from typing import Optional
import io
import statsmodels.api as sm
from statsmodels.base.model import Results
from statsmodels.base.wrapper import ResultsWrapper
from sklearn.base import BaseEstimator
from pyautocausal.persistence.parameter_mapper import make_transformable
from typing import Any
from pyautocausal.pipelines.library.specifications import BaseSpec
from linearmodels.shared.base import _SummaryStr


@make_transformable
def write_linear_models_to_summary(res: BaseSpec) -> str:
    # Handle the case where res is a BaseSpec object with a model attribute
    if hasattr(res, 'model'):
        result = res.model
    else:
        raise ValueError("res must be a BaseSpec object with a model attribute")
    if not isinstance(result, _SummaryStr):
        raise ValueError("res must be a BaseSpec object with a model attribute that is a (linearmodels) _SummaryStr object")
    
    # Create summary
    try:
        buffer = io.StringIO()
        buffer.write(str(result))
        return buffer.getvalue()
    except Exception as e:
        return f"Error creating summary: {str(e)}"
    
@make_transformable
def write_statsmodels_to_summary(res: BaseSpec) -> str:
    # Handle the case where res is a BaseSpec object with a model attribute
    if hasattr(res, 'model'):
        result = res.model
    else:
        raise ValueError("res must be a BaseSpec object with a model attribute")
    if not isinstance(result, Results) and not isinstance(result, ResultsWrapper):
        raise ValueError("res must be a BaseSpec object with a model attribute that is a (statsmodels) Results or ResultsWrapper object")
    
    # Create summary
    try:
        buffer = io.StringIO()
        buffer.write(str(result.summary()))
        return buffer.getvalue()
    except Exception as e:
        return f"Error creating summary: {str(e)}"

@make_transformable
def write_sklearn_summary(res: BaseEstimator) -> str:
    output = []
    output.append(f"Model Type: {type(res).__name__}")
    output.append("\nModel Parameters:")
    output.append(str(res.get_params()))
    
    if hasattr(res, 'classes_'):
        output.append("\nClasses:")
        output.append(str(res.classes_))
        
    if hasattr(res, 'feature_importances_'):
        output.append("\nFeature Importances:")
        output.append(str(res.feature_importances_))
        
    if hasattr(res, 'coef_'):
        output.append("\nCoefficients:")
        output.append(str(res.coef_))
        
    if hasattr(res, 'score'):
        output.append("\nModel Score:")
        output.append(str(res.score))
        
    return '\n'.join(output)


def write_statsmodels_summary_notebook(output: Any) -> None:
    print(output)

def display_cs_results_notebook(output: Any) -> None:
    """Display Callaway & Sant'Anna results in a notebook."""
    print(output)
