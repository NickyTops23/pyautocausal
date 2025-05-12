from typing import Optional
import io
import statsmodels.api as sm
from statsmodels.base.model import Results
from sklearn.base import BaseEstimator
from pyautocausal.persistence.parameter_mapper import make_transformable
from typing import Any

@make_transformable
def write_statsmodels_summary(res: Any) -> str:
    # Handle the case where res is a BaseSpec object with a model attribute
    if hasattr(res, 'model'):
        result = res.model
    elif isinstance(res, Results):
        # Assume res is already a statsmodels Results object
        result = res
    else:
        # For statsmodels OLS objects that aren't fitted, return type info
        import statsmodels.regression.linear_model as lm
        if isinstance(res, lm.OLS):
            return f"OLS model (unfitted): {type(res).__name__}\nThis model needs to be fitted before a summary can be generated."
        result = res
    
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