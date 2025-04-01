import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import Optional, Any, Union, Dict, List, Tuple
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from .output import StatsmodelsOutputAction
from .base_estimator import BaseEstimator
from .specifications import (
    StandardSpecification, 
    DiDSpecification, 
    EventStudySpecification,
    StaggeredDiDSpecification,
    TreatmentType
)
from .specifications import ContinuousTreatmentSpecification
from pyautocausal.persistence.parameter_mapper import make_transformable


class OLS(BaseEstimator):
    """Ordinary Least Squares regression analysis for treatment effect estimation."""
    
    @classmethod
    @make_transformable
    def action(cls, inputs: Dict[str, Any], **kwargs) -> sm.OLS:
        """
        Estimate treatment effect using OLS regression.
        
        Args:
            inputs: Dictionary containing:
                - data: DataFrame with outcome and treatment variables
                - specification: Model specification dictionary
            **kwargs: Additional arguments passed to statsmodels OLS
            
        Returns:
            Fitted statsmodels OLS model object
        """
        # Extract data and specification from inputs
        data = inputs.get('data')
        weights = kwargs.get('weights')
        
        
        # Get data from specification
        formula = inputs.get('formula')
                
        # Fit the model
        if weights is not None:
            model = sm.WLS.from_formula(formula, data=data, weights=weights).fit(cov_type='HC1')
        else:
            model = sm.OLS.from_formula(formula, data=data).fit(cov_type='HC1')
                
        return model
    
    @classmethod
    @make_transformable
    def output(cls, model: sm.OLS) -> str:
        """Format the OLS model results."""
        return StatsmodelsOutputAction(model)


class WOLS(OLS):
    """Weighted Ordinary Least Squares regression."""
    
    @classmethod
    @make_transformable
    def action(cls, inputs: Dict[str, Any], **kwargs) -> sm.WLS:
        """
        Estimate treatment effect using Weighted OLS regression.
        
        Args:
            inputs: Dictionary containing:
                - data: DataFrame with outcome and treatment variables
                - specification: Model specification dictionary
                - weights: sample weights
        Returns:
            Fitted statsmodels WLS model object
        """
        weights = inputs.get('weights')
        if weights is None:
            raise ValueError("Weights must be provided for WOLS")
        
        # Ensure weights are valid (no NaN, inf, or zero sum)
        if np.isnan(weights).any() or np.isinf(weights).any() or np.sum(weights) == 0:
            # In case of invalid weights, use uniform weights as fallback
            weights = np.ones_like(weights) / len(weights)
            inputs['weights'] = weights
            
        # Ensure weights are properly scaled
        if np.sum(weights) != 1.0 and np.sum(weights) != 0.0:
            weights = weights / np.sum(weights)
            inputs['weights'] = weights
            
        return super().action(inputs, **kwargs)





class DoubleLasso(BaseEstimator):
    """Double/Debiased Lasso estimation following Chernozhukov et al. (2018)."""
    
    @classmethod
    @make_transformable
    def action(cls, inputs: Dict[str, Any], **kwargs) -> sm.OLS:
        """
        Estimate treatment effect using Double/Debiased Lasso.
        
        Implementation follows the algorithm from:
        Chernozhukov et al. (2018) - Double/Debiased Machine Learning
        
        Args:
            inputs: Dictionary containing:
                - data: DataFrame with outcome and treatment variables
                - specification: Model specification dictionary
            **kwargs: Additional arguments including:
                - weights: Optional sample weights
                - alpha: Regularization strength for Lasso
                - cv_folds: Number of cross-validation folds
                
        Returns:
            Fitted OLS model for the final stage regression
        """
        # Extract data and specification from inputs
        data = inputs.get('data')
        specification = inputs.get('specification')
        
        # Extract optional parameters from kwargs
        alpha = kwargs.get('alpha', 0.1)
        cv_folds = kwargs.get('cv_folds', 5)
        
        # If specification isn't provided, create a standard specification
        if specification is None:
            specification = StandardSpecification.create(data)
        
        # Get information from specification
        outcome_col = specification.get('outcome_col', 'y')
        treatment_col = specification.get('treatment_col', 'treat')
        control_cols = specification.get('control_cols', [])
        
        # Prepare data
        y = data[outcome_col]
        d = data[treatment_col]
        
        if not control_cols:
            # If no controls are specified, can't do double lasso
            raise ValueError("Control variables are required for double lasso")
            
        X = data[control_cols]
        
        # Initialize cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Arrays to store residuals
        y_resid = np.zeros_like(y)
        d_resid = np.zeros_like(d)
        
        # Cross-fitting
        for train_idx, test_idx in kf.split(X):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            d_train, d_test = d.iloc[train_idx], d.iloc[test_idx]
            
            # First stage: outcome equation
            lasso_y = Lasso(alpha=alpha, random_state=42)
            lasso_y.fit(X_train, y_train)
            y_pred = lasso_y.predict(X_test)
            y_resid[test_idx] = y_test - y_pred
            
            # First stage: treatment equation
            lasso_d = Lasso(alpha=alpha, random_state=42)
            lasso_d.fit(X_train, d_train)
            d_pred = lasso_d.predict(X_test)
            d_resid[test_idx] = d_test - d_pred
        
        # Second stage: treatment effect estimation
        final_model = sm.OLS(
            y_resid,
            sm.add_constant(pd.Series(d_resid, name=treatment_col))
        ).fit(cov_type='HC1')
        
        return final_model
    
    @classmethod
    @make_transformable
    def output(cls, model: sm.OLS) -> str:
        """Format the Double Lasso model results."""
        return StatsmodelsOutputAction(model)

