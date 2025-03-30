from typing import Optional
import io
import statsmodels.api as sm
from sklearn.base import BaseEstimator



def StatsmodelsOutputAction(model: BaseEstimator) -> str:
    buffer = io.StringIO()
    buffer.write(str(model.summary()))
    return buffer.getvalue()

def ScikitLearnOutputAction(model: BaseEstimator) -> str:
    output = []
    output.append(f"Model Type: {type(model).__name__}")
    output.append("\nModel Parameters:")
    output.append(str(model.get_params()))
    
    if hasattr(model, 'classes_'):
        output.append("\nClasses:")
        output.append(str(model.classes_))
        
    if hasattr(model, 'feature_importances_'):
        output.append("\nFeature Importances:")
        output.append(str(model.feature_importances_))
        
    if hasattr(model, 'coef_'):
        output.append("\nCoefficients:")
        output.append(str(model.coef_))
        
    if hasattr(model, 'score'):
        output.append("\nModel Score:")
        output.append(str(model.score))
        
    return '\n'.join(output)



