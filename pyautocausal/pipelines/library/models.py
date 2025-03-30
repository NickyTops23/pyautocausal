import pandas as pd
import statsmodels.api as sm
import io
from typing import Callable, Optional
from abc import abstractmethod
from pyautocausal.orchestration.nodes import Node, ExecutableGraph, OutputConfig
from pyautocausal.persistence.output_config import OutputType
from .library import LibraryNode
from .output import StatsmodelsOutputAction
from pyautocausal.orchestration.condition import Condition
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
import numpy as np

class PassthroughNode(LibraryNode):
    """Node that simply passes data through without modification."""
    
    def __init__(self, 
                 name: str = "Passthrough", 
                 graph: Optional[ExecutableGraph] = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = False,
                 output_filename: str = "passthrough"):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            output_filename=output_filename,
            graph=graph
        )
   
    @staticmethod
    def action(df: pd.DataFrame) -> pd.DataFrame:
        """Simply return the input DataFrame unchanged."""
        return df

class OLSNode(LibraryNode):
    """Node for Ordinary Least Squares regression analysis."""
    
    def __init__(self, 
                 name: str = "OLS Treatment Effect", 
                 graph: Optional[ExecutableGraph] = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = True,
                 output_filename: str = "ols_treatment_effect"):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            output_filename=output_filename,
            graph=graph
        )
   
    @staticmethod
    def action(df: pd.DataFrame) -> sm.OLS:
        """Estimate treatment effect using OLS regression."""
        required_columns = ['y', 'treat']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        if not pd.api.types.is_numeric_dtype(df['y']) or not pd.api.types.is_numeric_dtype(df['treat']):
            raise TypeError("'y' and 'treat' must be numeric types.")
        
        # Make a copy of the dataframe and drop rows with NaN values
        df = df.copy()
        df = df.dropna()
        
        y = df['y']
        
        # Select only numeric columns and exclude non-numeric ones
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Exclude outcome variable but keep treatment
        feature_cols = [col for col in numeric_cols if col != 'y']
        
        # Create matrix with only numeric features
        X = df[feature_cols]
        X = sm.add_constant(X)
        
        # Check for any remaining NaN values
        if X.isna().any().any() or y.isna().any():
            raise ValueError("There are still NaN values in the data after preprocessing")

        model = sm.OLS(y, X).fit(cov_type='HC1')  # Added robust standard errors
                
        return model
        
    @staticmethod
    def condition() -> Condition:
        """Default condition checking if sample size is appropriate for OLS."""
        return Condition(
            lambda df: len(df) <= 100,
            "Sample size is less than or equal to 100 observations"
        )
    
    @staticmethod
    def output(model) -> str:
        """Return a function that formats the OLS model results."""
        return StatsmodelsOutputAction(model)


class DoubleLassoNode(LibraryNode):
    """Node for Double/Debiased Lasso estimation following Chernozhukov et al. (2018)."""
    
    def __init__(self, 
                 name: str = "Double Lasso Treatment Effect", 
                 graph: Optional[ExecutableGraph] = None,   
                 condition: Optional[Condition] = None,
                 save_node: bool = True,
                 output_filename: str = "double_lasso_effect",
                 alpha: float = 0.1,
                 cv_folds: int = 5):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            output_filename=output_filename,
            graph=graph
        )
        self.alpha = alpha
        self.cv_folds = cv_folds

    @staticmethod
    def action(df: pd.DataFrame) -> sm.OLS:
        """
        Estimate treatment effect using Double/Debiased Lasso.
        
        Implementation follows the algorithm from:
        Chernozhukov et al. (2018) - Double/Debiased Machine Learning
        
        Args:
            df: DataFrame with columns:
                - y: Outcome variable
                - treat: Treatment indicator
                - Additional columns used as controls
                
        Returns:
            Fitted OLS model for the final stage regression
        """
        required_columns = ['y', 'treat']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Get features (exclude treatment and outcome)
        exclude_cols = ['y', 'treat']
        features = [col for col in df.columns if col not in exclude_cols]
        
        if not features:
            raise ValueError("No features available for double lasso")
            
        # Prepare data
        y = df['y']
        d = df['treat']
        X = df[features]
        
        # Initialize cross-validation
        kf = KFold(shuffle=True, random_state=42)
        
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
            lasso_y = Lasso(alpha=0.1, random_state=42)
            lasso_y.fit(X_train, y_train)
            y_pred = lasso_y.predict(X_test)
            y_resid[test_idx] = y_test - y_pred
            
            # First stage: treatment equation
            lasso_d = Lasso(alpha=0.1, random_state=42)
            lasso_d.fit(X_train, d_train)
            d_pred = lasso_d.predict(X_test)
            d_resid[test_idx] = d_test - d_pred
        
        # Second stage: treatment effect estimation
        final_model = sm.OLS(
            y_resid,
            sm.add_constant(pd.Series(d_resid, name='treat'))
        ).fit(cov_type='HC1')  # Heteroskedasticity-robust standard errors
        
        return final_model

    @staticmethod
    def condition() -> Condition:
        """Default condition checking if sample size is appropriate for Double Lasso."""
        def check_requirements(df: pd.DataFrame) -> bool:
            # Check sample size relative to number of features
            n_features = len(df.columns) - 2  # Excluding y and treat
            min_samples = max(100, 4 * n_features)  # Rule of thumb
            return len(df) >= min_samples
        
        return Condition(
            check_requirements,
            "Sample size must be sufficient relative to number of features"
        )

    @staticmethod
    def output(model) -> str:
        """Return a function that formats the Double Lasso model results."""
        return ScikitLearnOutputAction(model)

class CovariateSelectionNode(LibraryNode):
    """Node for control variable selection."""
    
    def __init__(self, 
                 name: str = "Control Variable Selection", 
                 graph: Optional[ExecutableGraph] = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = True,
                 output_filename: str = "control_variable_selection"):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            output_filename=output_filename,
            graph=graph
        )
    
    @staticmethod
    def action(df: pd.DataFrame) -> list[str]:
        """
        Select control variables from the dataset.
        
        Args:
            df: DataFrame with potential control variables
            
        Returns:
            List of column names selected as controls
        """
        # Exclude standard non-control columns
        exclude_cols = ['y', 'treat', 't', 'id_unit', 'post', 'relative_time']
        
        # Get numeric columns that aren't in exclude_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        controls = [col for col in numeric_cols if col not in exclude_cols]
        
        if not controls:
            raise ValueError("No numeric control variables found in dataset")
            
        return controls

class PSMNode(LibraryNode):
    """Node for Propensity Score Matching analysis."""
    
    caliper = None
    n_neighbors = 1
    
    def __init__(self, 
                 name: str = "PSM Treatment Effect", 
                 graph: Optional[ExecutableGraph] = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = True,
                 output_filename: str = "psm_treatment_effect",
                 n_neighbors: int = 1,
                 caliper: Optional[float] = None):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            output_filename=output_filename,
            graph=graph
        )
        # Set class attributes
        PSMNode.n_neighbors = n_neighbors
        PSMNode.caliper = caliper

    @staticmethod
    def action(df: pd.DataFrame, covariates: list[str] = None, n_neighbors: int = 1, caliper: Optional[float] = None) -> tuple[sm.OLS, np.ndarray]:
        """
        Perform propensity score matching and estimate treatment effect.
        
        Args:
            df: DataFrame with 'treat' column and covariates
            covariates: List of column names to use as covariates
            n_neighbors: Number of neighbors for matching
            caliper: Maximum distance for matching
            
        Returns:
            Tuple containing:
            - Fitted OLS model object estimating treatment effect with matched data
            - Propensity scores
        """
        required_columns = ['treat', 'y']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Make a copy of the dataframe and drop rows with NaN values
        df = df.copy()
        
        # Get features (exclude treatment and outcome if present)
        exclude_cols = ['treat', 'y', 't', 'id_unit', 'post', 'relative_time']
        
        # Select only numeric columns to avoid issues with objects
        numeric_cols = df.select_dtypes(include=['number']).columns
        features = [col for col in numeric_cols if col not in exclude_cols]
        
        if not features:
            raise ValueError("No features available for matching")
        
        # Only keep necessary columns
        cols_to_keep = required_columns + features
        df = df[cols_to_keep]
        
        # Drop rows with NaN values
        df = df.dropna()
            
        X = df[features]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X_scaled, df['treat'])
        ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Add propensity scores to dataframe
        df_matched = df.copy()
        df_matched['ps_score'] = ps_scores
        
        # Add an index column we can reference later
        df_matched = df_matched.reset_index(drop=True)
        
        # Perform matching
        treated = df_matched[df_matched['treat'] == 1]
        control = df_matched[df_matched['treat'] == 0]
        
        # Initialize matching algorithm
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(control[['ps_score']])
        
        # Find matches
        distances, indices = nbrs.kneighbors(treated[['ps_score']])
        
        # Apply caliper if specified
        if caliper is not None:
            valid_matches = distances.ravel() <= caliper
            treated = treated[valid_matches]
            indices = indices[valid_matches]
            
        # Get matched control units
        matched_control_indices = indices.ravel()
        
        # Mark matched units and assign match IDs
        df_matched['matched'] = False
        df_matched['match_id'] = -1
        
        for i, (treat_idx, _) in enumerate(treated.iterrows()):
            control_indices = indices[i]
            match_id = i
            
            # Mark treated unit
            df_matched.loc[treat_idx, 'matched'] = True
            df_matched.loc[treat_idx, 'match_id'] = match_id
            
            # Mark control units
            for control_idx in control_indices:
                df_matched.loc[control.index[control_idx], 'matched'] = True
                df_matched.loc[control.index[control_idx], 'match_id'] = match_id
        
        # Keep only matched observations for treatment effect estimation
        matched_data = df_matched[df_matched['matched'] == True]
        
        # Estimate treatment effect using OLS
        y = matched_data['y']
        X = sm.add_constant(matched_data[['treat']])
        
        # Run regression with robust standard errors
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        return (model, ps_scores)

    @staticmethod
    def condition() -> Condition:
        """Default condition checking if sample is imbalanced."""
        return Condition(
            lambda df: True,  # We'll use the check_imbalance condition from the graph
            "Always allow PSM node execution"
        )

    @staticmethod
    def output(psm_output: tuple[sm.OLS, np.ndarray]) -> str:
        """Return a function that formats the PSM model results."""
        return StatsmodelsOutputAction(psm_output[0])

class DiDNode(LibraryNode):
    """Node for Two-Period Difference-in-Differences analysis."""
    
    def __init__(self, 
                 name: str = "DiD Treatment Effect",
                 graph: Optional[ExecutableGraph] = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = True,
                 output_filename: str = "did_treatment_effect",
                 use_unit_fe: bool = True,
                 use_time_fe: bool = True):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            output_filename=output_filename,
            graph=graph
        )
        self.use_unit_fe = use_unit_fe
        self.use_time_fe = use_time_fe

    @staticmethod
    def action(df: pd.DataFrame) -> sm.OLS:
        """
        Estimate treatment effect using Difference-in-Differences.
        
        Args:
            df: DataFrame with columns:
                - treat: Treatment indicator
                - t: Time period
                - id_unit: Unit identifier
                - y: Outcome variable
                - post: Post-treatment period indicator
                
        Returns:
            String containing regression results
        """
        required_columns = ['treat', 't', 'id_unit', 'y', 'post']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Make a simpler implementation that uses only the treatment*post interaction
        # and removes all problematic data
        
        # Create a clean copy with only the columns we absolutely need
        clean_df = df[required_columns].copy()
        
        # Drop rows with NaN values
        clean_df = clean_df.dropna()
        
        # Create the DiD interaction term
        clean_df['treat_post'] = clean_df['treat'] * clean_df['post']
        
        # Prepare the regression variables
        y = clean_df['y']
        X = clean_df[['treat', 'post', 'treat_post']]
        X = sm.add_constant(X)
        
        # Ensure all data is numeric
        for col in X.columns:
            X[col] = X[col].astype(float)
        y = y.astype(float)
        
        # Run the regression with robust standard errors
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        return model

    @staticmethod
    def condition() -> Condition:
        """Default condition checking requirements for DiD."""
        def check_did_requirements(df: pd.DataFrame) -> bool:
            if not all(col in df.columns for col in ['treat', 't', 'id_unit', 'post']):
                return False
            # Check if we have both treated and control units
            has_treated = (df['treat'] == 1).any()
            has_control = (df['treat'] == 0).any()
            # Check if we have both pre and post periods
            has_pre = (df['post'] == 0).any()
            has_post = (df['post'] == 1).any()
            return has_treated and has_control and has_pre and has_post
        
        return Condition(
            check_did_requirements,
            "Data must have treated and control units, and pre/post periods"
        )
    def output(model) -> str:
        """Return a function that formats the DiD model results."""
        return StatsmodelsOutputAction(model)

class EventStudyNode(LibraryNode):
    """Node for Event Study analysis."""
    
    def __init__(self, 
                 name: str = "Event Study Treatment Effect",
                 graph: Optional[ExecutableGraph] = None,
                 condition: Optional[Condition] = None,
                 save_node: bool = True):
        super().__init__(
            name=name,
            condition=condition,
            save_node=save_node,
            graph=graph
        )
    
    @staticmethod
    def action(df: pd.DataFrame) -> sm.OLS:
        """
        Estimate event study coefficients.
        
        Args:
            df: DataFrame with columns:
                - treat: Treatment indicator
                - t: Time period
                - id_unit: Unit identifier
                - y: Outcome variable
                - relative_time: Time relative to treatment
                
        Returns:
            Fitted OLS model object with event study coefficients
        """
        required_columns = ['treat', 't', 'id_unit', 'y', 'relative_time']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Create event time dummies (excluding one period for identification)
        df = pd.get_dummies(df, columns=['relative_time'], prefix='event')
        event_cols = [col for col in df.columns if col.startswith('event_')]
        if len(event_cols) > 1:
            event_cols = event_cols[1:]  # Drop one period
            
        # Create unit fixed effects
        df = pd.get_dummies(df, columns=['id_unit'], prefix='unit')
        unit_cols = [col for col in df.columns if col.startswith('unit_')]
        
        # Prepare regression variables
        y = df['y']
        X = df[event_cols + unit_cols]
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                cov_kwds={'groups': df['id_unit']})
        
        return model
        