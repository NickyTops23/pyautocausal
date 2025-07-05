import pandas as pd
import pytest
from pyautocausal.data_validation.checks.causal_checks import (
    BinaryTreatmentCheck,
    PanelStructureCheck,
    BinaryTreatmentConfig,
    PanelStructureConfig,
)


def test_binary_treatment_check_valid():
    """Tests that the binary treatment check passes with valid data."""
    df = pd.DataFrame({"treat": [0, 1, 0, 1, 1]})
    check = BinaryTreatmentCheck(config=BinaryTreatmentConfig(treatment_column="treat"))
    result = check.validate(df)
    assert result.passed


def test_binary_treatment_check_invalid():
    """Tests that the binary treatment check fails with invalid data."""
    df = pd.DataFrame({"treat": [0, 1, 2, 0, -1]})
    check = BinaryTreatmentCheck(config=BinaryTreatmentConfig(treatment_column="treat"))
    result = check.validate(df)
    assert not result.passed
    assert "invalid values" in result.issues[0].message


def test_panel_structure_check_valid():
    """Tests that the panel structure check passes with a balanced panel."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3],
            "time": [2000, 2001, 2000, 2001, 2000, 2001],
            "value": [10, 12, 20, 22, 30, 32],
        }
    )
    check = PanelStructureCheck(config=PanelStructureConfig(unit_column="unit", time_column="time"))
    result = check.validate(df)
    assert result.passed


def test_panel_structure_check_unbalanced():
    """Tests that the panel structure check fails with an unbalanced panel."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 3, 3],
            "time": [2000, 2001, 2000, 2000, 2001],
            "value": [10, 12, 20, 30, 32],
        }
    )
    check = PanelStructureCheck(config=PanelStructureConfig(unit_column="unit", time_column="time"))
    result = check.validate(df)
    assert not result.passed
    assert "Panel is unbalanced" in result.issues[0].message 