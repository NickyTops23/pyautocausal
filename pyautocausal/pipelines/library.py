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
            - Additional columns used as covariates

    Returns:
        str: Formatted summary of the Double ML estimation results
    """
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