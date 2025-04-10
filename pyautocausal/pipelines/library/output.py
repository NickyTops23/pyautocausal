from typing import Optional
import io
import statsmodels.api as sm
from statsmodels.base.model import Results
from sklearn.base import BaseEstimator
from pyautocausal.persistence.parameter_mapper import make_transformable

@make_transformable
def write_statsmodels_summary(res: Results ) -> str:
    buffer = io.StringIO()
    
    try:
        # First try to get the standard summary
        buffer.write(str(res.summary()))
    except Exception as e:
        # If the standard summary fails, try to create a simplified summary
        buffer.write(f"Error generating standard model summary: {str(e)}\n\n")
        buffer.write(f"Model Type: {type(res).__name__}\n\n")
        
        # Basic model parameters that are likely to be available
        buffer.write("Basic Model Information:\n")
        if hasattr(res, 'params'):
            buffer.write("\nParameters:\n")
            buffer.write(str(res.params))
            
        if hasattr(res, 'pvalues'):
            buffer.write("\n\nP-values:\n")
            try:
                buffer.write(str(res.pvalues))
            except:
                buffer.write("Unable to generate p-values")
                
        if hasattr(res, 'rsquared'):
            try:
                buffer.write(f"\n\nR-squared: {res.rsquared}")
            except:
                pass
                
        if hasattr(res, 'nobs'):
            buffer.write(f"\n\nObservations: {res.nobs}")
                
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



