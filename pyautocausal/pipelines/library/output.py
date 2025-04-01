from typing import Optional
import io
import statsmodels.api as sm
from sklearn.base import BaseEstimator



def StatsmodelsOutputAction(model: BaseEstimator) -> str:
    buffer = io.StringIO()
    
    try:
        # First try to get the standard summary
        buffer.write(str(model.summary()))
    except Exception as e:
        # If the standard summary fails, try to create a simplified summary
        buffer.write(f"Error generating standard model summary: {str(e)}\n\n")
        buffer.write(f"Model Type: {type(model).__name__}\n\n")
        
        # Basic model parameters that are likely to be available
        buffer.write("Basic Model Information:\n")
        if hasattr(model, 'params'):
            buffer.write("\nParameters:\n")
            buffer.write(str(model.params))
            
        if hasattr(model, 'pvalues'):
            buffer.write("\n\nP-values:\n")
            try:
                buffer.write(str(model.pvalues))
            except:
                buffer.write("Unable to generate p-values")
                
        if hasattr(model, 'rsquared'):
            try:
                buffer.write(f"\n\nR-squared: {model.rsquared}")
            except:
                pass
                
        if hasattr(model, 'nobs'):
            buffer.write(f"\n\nObservations: {model.nobs}")
                
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



