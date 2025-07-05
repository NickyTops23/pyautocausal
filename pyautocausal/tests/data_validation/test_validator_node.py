import pandas as pd
import pytest
from pyautocausal.data_cleaning.hints import ConvertToCategoricalHint
from pyautocausal.data_validation.base import DataValidationResult, DataValidationCheck, ValidationIssue, ValidationSeverity, DataValidationConfig
from pyautocausal.data_validation.validator_node import DataValidator, AggregatedValidationResult


class MockCheck(DataValidationCheck):
    def __init__(self, is_valid, name="mock_check", errors=None, hints=None):
        super().__init__(config=DataValidationConfig())
        self._is_valid = is_valid
        self._name = name
        self._errors = errors or []
        self._hints = hints or []

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def get_default_config(cls) -> DataValidationConfig:
        return DataValidationConfig()

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = [ValidationIssue(severity=ValidationSeverity.ERROR, message=e) for e in self._errors]
        return self._create_result(passed=self._is_valid, issues=issues, cleaning_hints=self._hints)


def test_data_validator_node_all_pass():
    """Tests that the validator node passes when all checks pass."""
    df = pd.DataFrame()
    checks = [
        MockCheck(is_valid=True, hints=[ConvertToCategoricalHint(target_columns=["A"], threshold=10)]),
        MockCheck(is_valid=True, hints=[ConvertToCategoricalHint(target_columns=["B"], threshold=10)]),
    ]
    node = DataValidator(checks=checks)
    result = node.validate(df)

    assert result.passed
    assert not result.get_all_issues()
    
    all_hints = [h for res in result.individual_results for h in res.cleaning_hints]
    assert len(all_hints) == 2
    assert isinstance(all_hints[0], ConvertToCategoricalHint)


def test_data_validator_node_one_fails():
    """Tests that the validator node fails if any check fails."""
    df = pd.DataFrame()
    checks = [
        MockCheck(is_valid=True, hints=[ConvertToCategoricalHint(target_columns=["A"], threshold=10)]),
        MockCheck(is_valid=False, errors=["Column C is bad"]),
    ]
    node = DataValidator(checks=checks)
    result = node.validate(df)

    assert not result.passed
    errors = result.get_all_issues()
    assert len(errors) == 1
    assert "Column C is bad" in errors[0].message
    
    all_hints = [h for res in result.individual_results for h in res.cleaning_hints]
    assert len(all_hints) == 1  # Should still collect hints from passing checks


def test_data_validator_node_no_checks():
    """Tests that the validator node passes if no checks are provided."""
    df = pd.DataFrame()
    node = DataValidator(checks=[])
    result = node.validate(df)

    assert result.passed
    assert not result.get_all_issues()
    all_hints = [h for res in result.individual_results for h in res.cleaning_hints]
    assert not all_hints 