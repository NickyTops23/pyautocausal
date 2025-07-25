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


def create_core_decision_structure(graph: ExecutableGraph, abs_text_dir: Path) -> None:
    """Create the core decision structure that routes data through different analysis paths.
    
    This is the heart of the causal inference pipeline. The decision flow is:
    
    1. Input data → Basic cleaning
    2. Multi-period check → Panel vs Cross-sectional path
    3. Panel path: Single unit check → Synthetic DiD vs Multi-unit methods
    4. Multi-unit path: Post-periods check → Event study vs Staggered DiD
    5. Staggered DiD: Never-treated check → CS-NT vs CS-NYT estimators
    
    Args:
        graph: The ExecutableGraph to add nodes to
        abs_text_dir: Directory for text outputs
    """
    
    # === INPUT AND CLEANING ===
    graph.create_input_node("df", input_dtype=pd.DataFrame)

    # Basic cleaning node - now uses logging-based approach
    def basic_clean_node(df: pd.DataFrame) -> pd.DataFrame:
        return create_basic_cleaner(df)
    
    graph.create_node('basic_cleaning', action_function=basic_clean_node, predecessors=["df"])
    
    def get_basic_cleaning_metadata(basic_cleaning: pd.DataFrame) -> str:
        # With logging-based approach, metadata would be captured from logs by the framework
        # For now, return a placeholder message
        return f"Basic Cleaning Summary:\nCleaned data has {len(basic_cleaning)} rows\nMetadata available in execution logs"
    
    graph.create_node(
        'basic_cleaning_metadata',
        action_function=get_basic_cleaning_metadata,
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'basic_cleaning_metadata',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["basic_cleaning"]
    )

    # === PRIMARY DECISION: PANEL VS CROSS-SECTIONAL ===
    graph.create_decision_node(
        'multi_period', 
        condition=has_multiple_periods.get_function(), 
        predecessors=["basic_cleaning"]
    )

    # === PANEL DATA PATH ===
    def panel_clean_node(multi_period: pd.DataFrame) -> pd.DataFrame:
        return create_panel_cleaner(multi_period)
    
    graph.create_node('panel_cleaned_data', action_function=panel_clean_node, predecessors=["multi_period"])
    
    def get_panel_cleaning_metadata(panel_cleaned_data: pd.DataFrame) -> str:
        # With logging-based approach, metadata would be captured from logs by the framework
        return f"Panel Cleaning Summary:\nCleaned data has {len(panel_cleaned_data)} rows\nMetadata available in execution logs"
    
    graph.create_node(
        'panel_cleaning_metadata',
        action_function=get_panel_cleaning_metadata,
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'panel_cleaning_metadata',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["panel_cleaned_data"]
    )

    # === CROSS-SECTIONAL DATA PATH ===
    def cross_sectional_clean_node(multi_period: pd.DataFrame) -> pd.DataFrame:
        return create_cross_sectional_cleaner(multi_period)
    
    graph.create_node('cross_sectional_cleaned_data', action_function=cross_sectional_clean_node, predecessors=["multi_period"])
    
    def get_cross_sectional_cleaning_metadata(cross_sectional_cleaned_data: pd.DataFrame) -> str:
        # With logging-based approach, metadata would be captured from logs by the framework
        return f"Cross-Sectional Cleaning Summary:\nCleaned data has {len(cross_sectional_cleaned_data)} rows\nMetadata available in execution logs"
    
    graph.create_node(
        'cross_sectional_cleaning_metadata',
        action_function=get_cross_sectional_cleaning_metadata,
        output_config=OutputConfig(
            output_filename=abs_text_dir / 'cross_sectional_cleaning_metadata',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["cross_sectional_cleaned_data"]
    )
    
    # === PANEL ANALYSIS DECISIONS ===
    
    # Decision 1: Single treated unit? → Synthetic DiD
    graph.create_decision_node(
        'single_treated_unit',
        condition=has_single_treated_unit.get_function(),
        predecessors=["panel_cleaned_data"]
    )
    
    # Decision 2: Multiple post periods? → Event study vs simple DiD  
    graph.create_decision_node(
        'multi_post_periods', 
        condition=has_minimum_post_periods.get_function(), 
        predecessors=["single_treated_unit"]
    )
    
    # Decision 3: Staggered treatment? → Staggered DiD vs standard event study
    graph.create_decision_node(
        'stag_treat', 
        condition=has_staggered_treatment.get_function(), 
        predecessors=["multi_post_periods"]
    )


def configure_core_decision_paths(graph: ExecutableGraph) -> None:
    """Configure the routing logic for the core decision structure.
    
    This defines which nodes are executed based on decision outcomes.
    The routing logic implements the causal inference decision tree.
    """
    
    # Primary routing: Panel vs Cross-sectional
    graph.when_false("multi_period", "cross_sectional_cleaned_data")
    graph.when_true("multi_period", "panel_cleaned_data")

    # Panel analysis routing
    graph.when_true("single_treated_unit", "synthdid_spec")      # → Synthetic DiD
    graph.when_false("single_treated_unit", "multi_post_periods") # → Multi-unit methods

    # Multi-unit routing  
    graph.when_true("multi_post_periods", "stag_treat")         # → Check for staggered treatment
    graph.when_false("multi_post_periods", "did_spec")         # → Simple DiD (not enough periods for event study)

    # Staggered treatment routing
    graph.when_true("stag_treat", "stag_spec")                 # → Staggered DiD methods
    graph.when_false("stag_treat", "event_spec")               # → Standard event study

    # Callaway & Sant'Anna method selection
    graph.when_true("has_never_treated", "cs_never_treated")   # → CS with never-treated control
    graph.when_false("has_never_treated", "cs_not_yet_treated") # → CS with not-yet-treated control 