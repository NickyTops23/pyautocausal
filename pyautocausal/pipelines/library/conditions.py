from typing import Callable, Union, Dict, Any
import pandas as pd
import numpy as np
from pyautocausal.persistence.parameter_mapper import make_transformable


def _get_dataframe(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """
    Helper function to extract DataFrame from input which can be either a DataFrame
    or a dictionary with a 'data' field containing a DataFrame.
    
    Args:
        df_or_dict: Either a DataFrame or a dictionary with a 'data' key
        
    Returns:
        pd.DataFrame: The extracted DataFrame
        
    Raises:
        ValueError: If input is a dict without a 'data' key or if 'data' is not a DataFrame
    """
    if isinstance(df_or_dict, pd.DataFrame):
        return df_or_dict
    elif isinstance(df_or_dict, dict) and 'data' in df_or_dict:
        if not isinstance(df_or_dict['data'], pd.DataFrame):
            raise ValueError("The 'data' field must be a pandas DataFrame")
        return df_or_dict['data']
    else:
        raise ValueError("Input must be either a DataFrame or a dictionary with a 'data' key")


def is_continuous(series: pd.Series) -> bool:
    """
    Check if a series contains continuous values (not just 0s and 1s).
    
    Args:
        series: pandas Series to check
        
    Returns:
        bool: True if series contains values other than 0 and 1
    """
    unique_values = series.unique()
    return (
        len(unique_values) > 2 or 
        (len(unique_values) == 2 and not all(v in [0, 1] for v in unique_values))
    )


def index_relative_time(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """Helper function to calculate relative time to treatment"""
    df = _get_dataframe(df_or_dict)
    
    if not all(col in df.columns for col in ['t', 'treat', 'id_unit']):
        raise ValueError("DataFrame must contain 't', 'treat', and 'id_unit' columns")
        
    # Find first treatment period for each unit
    treatment_starts = df[df['treat'] == 1].groupby('id_unit')['t'].min()
    
    # Create relative time column
    df = df.copy()
    df['relative_time'] = df.apply(
        lambda row: row['t'] - treatment_starts.get(row['id_unit'], np.inf)
        if row['id_unit'] in treatment_starts.index else -np.inf,
        axis=1
    )
    return df


class TimeConditions:
    """Functions for checking time-related characteristics of datasets"""
    
    @make_transformable
    @staticmethod
    def has_minimum_periods(df_or_dict: Union[pd.DataFrame, Dict[str, Any]], min_periods: int = 3) -> bool:
        """
        Check if dataset has at least the specified number of time periods.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 't' column
            min_periods: Minimum number of periods required
            
        Returns:
            bool: True if dataset has at least min_periods time periods
        """
        df = _get_dataframe(df_or_dict)
        if 't' not in df.columns:
            return False  # Single period case
        return df['t'].nunique() >= min_periods
    
    @make_transformable
    @staticmethod
    def has_multiple_periods(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Check if dataset has multiple time periods.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 't' column
            
        Returns:
            bool: True if dataset has multiple time periods
        """
        df = _get_dataframe(df_or_dict)
        if 't' not in df.columns:
            return False  # Single period case
        return df['t'].nunique() > 1
    
    @make_transformable
    @staticmethod
    def has_single_period(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Check if dataset has only a single time period.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 't' column
            
        Returns:
            bool: True if dataset has only a single time period
        """
        df = _get_dataframe(df_or_dict)
        if 't' not in df.columns:
            return True  # No time column means single period
        return df['t'].nunique() <= 1

    @make_transformable
    @staticmethod
    def has_staggered_treatment(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Check if treatment timing is staggered across units.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 'treat', 'id_unit', and 't' columns
            
        Returns:
            bool: True if treatment is staggered across units
        """
        df = _get_dataframe(df_or_dict)
        if not all(col in df.columns for col in ['treat', 'id_unit', 't']):
            return False
            
        # Check if treatment starts at different times for different units
        treatment_starts = df.groupby('id_unit')['treat'].apply(
            lambda x: x.eq(1).idxmax() if x.eq(1).any() else None
        )
        return len(treatment_starts.unique()) > 1
    
    @make_transformable
    @staticmethod
    def has_minimum_pre_periods(df_or_dict: Union[pd.DataFrame, Dict[str, Any]], min_periods: int = 3) -> bool:
        """
        Check if there are enough pre-treatment periods.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 't', 'treat', and 'id_unit' columns
            min_periods: Minimum number of pre-treatment periods required
            
        Returns:
            bool: True if dataset has at least min_periods pre-treatment periods
        """
        df = _get_dataframe(df_or_dict)
        if not all(col in df.columns for col in ['t', 'treat', 'id_unit']):
            return False
        df = index_relative_time(df)
        return df[df['relative_time'] < 0]['relative_time'].nunique() >= min_periods
    
    @make_transformable
    @staticmethod
    def has_minimum_post_periods(df_or_dict: Union[pd.DataFrame, Dict[str, Any]], min_periods: int = 3) -> bool:
        """
        Check if there are enough post-treatment periods.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 't', 'treat', and 'id_unit' columns
            min_periods: Minimum number of post-treatment periods required
            
        Returns:
            bool: True if dataset has at least min_periods post-treatment periods
        """
        df = _get_dataframe(df_or_dict)
        if not all(col in df.columns for col in ['t', 'treat', 'id_unit']):
            return False
        df = index_relative_time(df)
        return df[df['relative_time'] >= 0]['relative_time'].nunique() >= min_periods


class TreatmentConditions:
    """Functions for checking treatment-related characteristics of datasets"""
    
    @make_transformable
    @staticmethod
    def has_multiple_treated_units(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Check if dataset has multiple treated units.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 'treat' and 'id_unit' columns
            
        Returns:
            bool: True if dataset has multiple treated units
        """
        df = _get_dataframe(df_or_dict)
        return len(df[df['treat']==1]['id_unit'].unique()) > 1
    
    @make_transformable
    @staticmethod
    def has_covariate_imbalance(df_or_dict: Union[pd.DataFrame, Dict[str, Any]], threshold: float = 0.5, imbalance_threshold: float = 0.25) -> bool:
        """
        Check if there is covariate imbalance between treatment and control groups.
        
        Args:
            df_or_dict: DataFrame or dictionary with 'data' field containing a DataFrame with 'treat' column and covariates
            threshold: Treatment threshold for continuous treatment
            imbalance_threshold: Standardized difference threshold above which covariates are considered imbalanced
            
        Returns:
            bool: True if covariates are imbalanced
        """
        df = _get_dataframe(df_or_dict)
        if 'treat' not in df.columns:
            raise ValueError("DataFrame must contain 'treat' column")
                
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        covariates = [col for col in numeric_cols if col not in ['treat', 't', 'id_unit']]
        
        if not covariates:
            return False
        
        # Define treatment groups based on continuous or binary treatment
        if is_continuous(df['treat']):
            treated_mask = df['treat'] > threshold
        else:
            treated_mask = df['treat'] == 1
            
        for col in covariates:
            treated_mean = df[treated_mask][col].mean()
            control_mean = df[~treated_mask][col].mean()
            treated_var = df[treated_mask][col].var()
            control_var = df[~treated_mask][col].var()
            
            std_diff = abs(treated_mean - control_mean) / np.sqrt((treated_var + control_var) / 2)
            
            if std_diff > imbalance_threshold:
                return True
                
        return False



