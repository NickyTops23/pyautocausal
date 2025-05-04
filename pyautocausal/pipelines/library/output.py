from typing import Optional
import io
import statsmodels.api as sm
from statsmodels.base.model import Results
from sklearn.base import BaseEstimator
from pyautocausal.persistence.parameter_mapper import make_transformable
from typing import Any

@make_transformable
def write_statsmodels_summary(res: Results ) -> str:
    buffer = io.StringIO()

    buffer.write(str(res.summary()))
                
    return buffer.getvalue()

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


@make_transformable
def write_statsmodels_summary_notebook(output: Any) -> str:
    """Notebook-friendly version of write_statsmodels_summary that also saves results to a file"""    

    notebook_display_string = f"""res_PLACEHOLDER.summary()

with open('model_summary.txt', 'w') as f:
    f.write(str(node_name_PLACEHOLDER))

# Display the summary
print(node_name_PLACEHOLDER)
"""
    
    return notebook_display_string