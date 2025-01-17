import pandas as pd
import statsmodels.api as sm
import io

def ols_treatment_effect(df: pd.DataFrame) -> str:
    """
    Estimate the treatment effect using OLS regression.

    Args:
        df (pd.DataFrame): DataFrame containing:
            - y: Outcome variable
            - treat: Treatment indicator
            - Additional columns used as covariates
        
    Returns:
        str: Formatted summary of the OLS regression results
    """
    # Check for necessary columns
    required_columns = ['y', 'treat']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Check for mixed types in 'y' and 'treat'
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


def doubleML_treatment_effect(df: pd.DataFrame) -> str:
    """
    Estimate treatment effect using Double Machine Learning.

    Args:
        df (pd.DataFrame): DataFrame containing:
            - y: Outcome variable
            - t: Treatment indicator
            - Additional numericcolumns used as covariates

    Returns:
        str: Formatted summary of the Double ML estimation results
    """
    # Check for necessary columns
    required_columns = ['y', 'treat']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Check for mixed types in 'y' and 'treat'
    if not pd.api.types.is_numeric_dtype(df['y']) or not pd.api.types.is_numeric_dtype(df['treat']):
        raise TypeError("'y' and 'treat' must be numeric types.")
    
    y = df['y']
    t = df['treat']
    X = df.drop(columns=['y', 'treat'])
    
    # First stage: residualize treatment and outcome
    model_t = sm.OLS(t, sm.add_constant(X)).fit()
    model_y = sm.OLS(y, sm.add_constant(X)).fit()
    
    t_residual = t - model_t.predict(sm.add_constant(X))
    y_residual = y - model_y.predict(sm.add_constant(X))
    
    # Second stage: estimate treatment effect
    effect_model = sm.OLS(y_residual, sm.add_constant(pd.Series(t_residual, name='treat'))).fit()

    buffer = io.StringIO()
    buffer.write(str(effect_model.summary()))
    return buffer.getvalue()



def data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the data format and distribute it.

    Args:
        df (pd.DataFrame): DataFrame to validate and distribute

    Returns:
        pd.DataFrame: Validated and distributed DataFrame
    """
    # Check for necessary columns
    required_columns = ['y', 'treat']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Check for mixed types in 'y' and 'treat'
    if not pd.api.types.is_numeric_dtype(df['y']) or not pd.api.types.is_numeric_dtype(df['treat']):
        raise TypeError("'y' and 'treat' must be numeric types.")

    return df
