"""Core decision logic for the causal inference pipeline.

This module contains the essential decision structure that determines which 
causal analysis methods are applied based on data characteristics.
"""

from pathlib import Path

import pandas as pd

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.conditions import (
    has_multiple_periods, 
    has_staggered_treatment, 
    has_minimum_post_periods, 
    has_sufficient_never_treated_units, 
    has_single_treated_unit
)
from pyautocausal.data_cleaner_interface.autocleaner import AutoCleaner

from pyautocausal.persistence.output_config import OutputConfig, OutputType


def create_basic_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """Create and execute basic data cleaning for both cross-sectional and panel data."""
    autocleaner = (
        AutoCleaner()
        .check_required_columns(required_columns=["treat", "y"])
        .check_column_types(expected_types={"treat": int, "y": float})
        .check_binary_treatment(treatment_column="treat")
        .check_for_missing_data(strategy="drop_rows", check_columns=["treat", "y"])
        .infer_and_convert_categoricals(ignore_columns=["treat", "y", "t", "id_unit"])
        .drop_duplicates()
    )
    return autocleaner.clean(df)


def create_panel_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """Create and execute panel-specific data cleaning."""
    autocleaner = (
        AutoCleaner()
        .check_required_columns(required_columns=["t", "id_unit"])
        .check_column_types(expected_types={"t": int, "id_unit": int})
        .check_for_missing_data(strategy="drop_rows")
        .infer_and_convert_categoricals(ignore_columns=["treat", "y", "t", "id_unit"])
        .drop_duplicates()
    )
    return autocleaner.clean(df)


def create_cross_sectional_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """Create and execute cross-sectional specific data cleaning.""" 
    autocleaner = (
        AutoCleaner()
        .check_for_missing_data(strategy="drop_rows")
        .infer_and_convert_categoricals(ignore_columns=["treat", "y", "t", "id_unit"])
    )
    return autocleaner.clean(df)


def _create_shared_head(graph: ExecutableGraph):
    """Creates the shared input and basic cleaning nodes for any graph."""
    graph.create_input_node("df", input_dtype=pd.DataFrame)
    
    def basic_clean_node(df: pd.DataFrame) -> pd.DataFrame:
        return create_basic_cleaner(df)
    
    graph.create_node('basic_cleaning', action_function=basic_clean_node, predecessors=["df"])


def create_panel_decision_structure(graph: ExecutableGraph, abs_text_dir: Path, predecessor: str = "basic_cleaning") -> None:
    """Create the decision structure for the panel data path."""
    
    # === PANEL DATA PATH (assumes multi-period data) ===
    def panel_clean_node(data_input: pd.DataFrame) -> pd.DataFrame:
        return create_panel_cleaner(data_input)
    
    graph.create_node('panel_cleaned_data', action_function=panel_clean_node, predecessors=[predecessor])
    
    def get_panel_cleaning_metadata(panel_cleaned_data: pd.DataFrame) -> str:
        return f"Panel Cleaning Summary:\nCleaned data has {len(panel_cleaned_data)} rows\nMetadata available in execution logs"
    
    graph.create_node(
        'panel_cleaning_metadata',
        action_function=get_panel_cleaning_metadata,
        output_config=OutputConfig(
            output_filename=str(abs_text_dir / 'panel_cleaning_metadata'),
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["panel_cleaned_data"]
    )
    
    # === PANEL DATA COMPLEXITY DECISIONS ===
    graph.create_decision_node(
        'single_treated_unit', 
        condition=has_single_treated_unit.get_function(), 
        predecessors=["panel_cleaned_data"]
    )
    
    graph.create_decision_node(
        'multi_post_periods', 
        condition=has_minimum_post_periods.get_function(), 
        predecessors=["single_treated_unit"]
    )
    
    graph.create_decision_node(
        'stag_treat', 
        condition=has_staggered_treatment.get_function(), 
        predecessors=["multi_post_periods"]
    )


def create_cross_sectional_decision_structure(graph: ExecutableGraph, abs_text_dir: Path, predecessor: str = "basic_cleaning") -> None:
    """Create the decision structure for the cross-sectional data path."""

    # === CROSS-SECTIONAL DATA PATH (assumes non-multi-period data) ===
    def cross_sectional_clean_node(data_input: pd.DataFrame) -> pd.DataFrame:
        return create_cross_sectional_cleaner(data_input)
    
    graph.create_node('cross_sectional_cleaned_data', action_function=cross_sectional_clean_node, predecessors=[predecessor])
    
    def get_cross_sectional_cleaning_metadata(cross_sectional_cleaned_data: pd.DataFrame) -> str:
        return f"Cross-Sectional Cleaning Summary:\nCleaned data has {len(cross_sectional_cleaned_data)} rows\nMetadata available in execution logs"
    
    graph.create_node(
        'cross_sectional_cleaning_metadata',
        action_function=get_cross_sectional_cleaning_metadata,
        output_config=OutputConfig(
            output_filename=str(abs_text_dir / 'cross_sectional_cleaning_metadata'),
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["cross_sectional_cleaned_data"]
    )


def configure_panel_decision_paths(graph: ExecutableGraph) -> None:
    """Configure the routing logic for the panel data decision structure."""
    
    # Panel analysis routing
    graph.when_true("single_treated_unit", "synthdid_spec")
    graph.when_false("single_treated_unit", "multi_post_periods")

    # Multi-unit routing  
    graph.when_true("multi_post_periods", "stag_treat")
    graph.when_false("multi_post_periods", "did_spec")

    # Staggered treatment routing
    graph.when_true("stag_treat", "stag_spec")
    graph.when_false("stag_treat", "event_spec")

    # Callaway & Sant'Anna method selection
    graph.when_true("has_never_treated", "cs_never_treated")
    graph.when_false("has_never_treated", "cs_not_yet_treated")


def configure_cross_sectional_decision_paths(graph: ExecutableGraph) -> None:
    """Configure the routing logic for the cross-sectional decision structure."""
    
    # No routing needed - cross-sectional graph has linear flow from basic_cleaning to analysis
    pass 