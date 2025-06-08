from typing import Optional
import io
import statsmodels.api as sm
from statsmodels.base.model import Results
from statsmodels.base.wrapper import ResultsWrapper
from sklearn.base import BaseEstimator
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.persistence.output_config import OutputConfig, OutputType
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

@make_transformable
def write_hainmueller_summary(spec, output_config: Optional[OutputConfig] = None) -> str:
    """
    Write a text summary of Hainmueeller Synthetic Control results.
    
    Args:
        spec: A specification object with fitted Hainmueeller model
        output_config: Optional output configuration for saving
        
    Returns:
        String containing the formatted summary
    """
    
    
    model = spec.hainmueller_model
    
    # Check if we have a SyntheticControlMethods Synth object
    if not hasattr(model, 'original_data'):
        raise ValueError("Hainmueeller model must be a SyntheticControlMethods Synth object")
    
    # Use SyntheticControlMethods results
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("HAINMUEELLER SYNTHETIC CONTROL RESULTS")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    # Display weight DataFrame
    summary_lines.append("DONOR UNIT WEIGHTS:")
    summary_lines.append("-" * 20)
    weight_df_str = str(model.original_data.weight_df)
    summary_lines.append(weight_df_str)
    summary_lines.append("")
    
    # Display comparison DataFrame
    summary_lines.append("COMPARISON RESULTS:")
    summary_lines.append("-" * 20)
    comparison_df_str = str(model.original_data.comparison_df)
    summary_lines.append(comparison_df_str)
    summary_lines.append("")
    
    # Display penalty parameter if available
    summary_lines.append(f"Penalty parameter: {model.original_data.pen}")
    summary_lines.append("")
    
    # Display RMSPE results if placebo test was run
    summary_lines.append("PLACEBO TEST RESULTS (RMSPE):")
    summary_lines.append("-" * 30)
    rmspe_df_str = str(model.original_data.rmspe_df)
    summary_lines.append(rmspe_df_str)
    summary_lines.append("")

    summary_lines.append("=" * 60)
    
    # Join all lines
    summary_text = "\n".join(summary_lines)
    
    # Print for notebook display
    print(summary_text)
    
    # Save to file if output config provided
    if output_config:
        if output_config.output_type == OutputType.TEXT:
            with open(f"{output_config.output_filename}.txt", 'w') as f:
                f.write(summary_text)
    
    return summary_text
