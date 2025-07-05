"""Example usage of the data validation system in PyAutoCausal.

This module demonstrates how to use the composable data validation checks
within a PyAutoCausal graph.
"""

import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.data_validation import (
    DataValidator,
    DataValidatorConfig,
    # Basic checks
    NonEmptyDataCheck,
    RequiredColumnsCheck,
    ColumnTypesCheck,
    # Missing data checks
    MissingDataCheck,
    # Causal checks
    BinaryTreatmentCheck,
    TreatmentVariationCheck,
    PanelStructureCheck,
)
from pyautocausal.data_validation.checks.basic_checks import (
    RequiredColumnsConfig,
    ColumnTypesConfig,
)
from pyautocausal.data_validation.checks.missing_data_checks import (
    MissingDataConfig,
)
from pyautocausal.data_validation.checks.causal_checks import (
    BinaryTreatmentConfig,
    TreatmentVariationConfig,
)


def create_validation_graph() -> ExecutableGraph:
    """Create a graph with data validation as the first step."""
    
    # Configure individual validation checks
    validator_config = DataValidatorConfig(
        fail_on_error=True,  # Fail the node if any ERROR issues are found
        fail_on_warning=False,  # Don't fail on warnings
        aggregation_strategy="all",  # All checks must pass
        check_configs={
            "required_columns": RequiredColumnsConfig(
                required_columns=["unit_id", "time", "treatment", "outcome"],
                case_sensitive=True
            ),
            "column_types": ColumnTypesConfig(
                expected_types={
                    "unit_id": str,
                    "time": int,
                    "treatment": int,
                    "outcome": float
                },
                categorical_threshold=50
            ),
            "missing_data": MissingDataConfig(
                max_missing_fraction=0.05,  # Allow up to 5% missing per column
                check_columns=["treatment", "outcome"]  # Only check these columns
            ),
            "binary_treatment": BinaryTreatmentConfig(
                treatment_column="treatment",
                allow_missing=False
            ),
            "treatment_variation": TreatmentVariationConfig(
                treatment_column="treatment",
                min_treated_fraction=0.1,
                max_treated_fraction=0.9,
                min_treated_count=20,
                min_control_count=20
            ),
        }
    )
    
    # Create the validator node with selected checks
    validator = DataValidator(
        checks=[
            NonEmptyDataCheck,
            RequiredColumnsCheck,
            ColumnTypesCheck,
            MissingDataCheck,
            BinaryTreatmentCheck,
            TreatmentVariationCheck,
            PanelStructureCheck,
        ],
        config=validator_config
    )
    
    # Create the graph
    graph = ExecutableGraph()
    graph.configure_runtime(output_path="./validation_outputs")
    
    # Input node for data
    graph.create_input_node("df", input_dtype=pd.DataFrame)
    
    # Validation node
    graph.create_node(
        "validate_data",
        action_function=validator,
        predecessors=["df"],
        output_config=OutputConfig(
            output_filename="validation_report",
            output_type=OutputType.JSON
        ),
        save_node=True
    )
    
    # Decision node based on validation
    graph.create_decision_node(
        "validation_passed",
        condition=lambda validate_data: validate_data.passed,
        predecessors=["validate_data"]
    )
    
    # Continue with analysis if validation passes
    graph.create_node(
        "analyze_data",
        action_function=lambda df: {"message": "Data validation passed, proceeding with analysis"},
        predecessors=["df"]
    )
    
    # Create error report if validation fails
    graph.create_node(
        "create_error_report",
        action_function=lambda validate_data: {
            "validation_failed": True,
            "summary": validate_data.summary,
            "failed_checks": validate_data.get_failed_checks(),
            "all_issues": [
                {
                    "check": issue.check_name,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "details": issue.details
                }
                for result in validate_data.individual_results
                for issue in result.issues
            ]
        },
        predecessors=["validate_data"],
        output_config=OutputConfig(
            output_filename="validation_errors",
            output_type=OutputType.JSON
        ),
        save_node=True
    )
    
    # Wire the decision node
    graph.when_true("validation_passed", "analyze_data")
    graph.when_false("validation_passed", "create_error_report")
    
    return graph


def create_minimal_validator() -> DataValidator:
    """Create a minimal validator for quick checks."""
    
    config = DataValidatorConfig(
        fail_on_error=True,
        check_configs={
            "required_columns": RequiredColumnsConfig(
                required_columns=["treatment", "outcome"]
            ),
            "binary_treatment": BinaryTreatmentConfig(
                treatment_column="treatment"
            )
        }
    )
    
    return DataValidator(
        checks=[RequiredColumnsCheck, BinaryTreatmentCheck],
        config=config
    )


def validate_dataframe_standalone(df: pd.DataFrame) -> None:
    """Example of using the validator outside of a graph context."""
    
    # Create a validator
    validator = DataValidator(
        checks=[
            NonEmptyDataCheck,
            RequiredColumnsCheck,
            BinaryTreatmentCheck,
            TreatmentVariationCheck,
        ],
        config=DataValidatorConfig(
            check_configs={
                "required_columns": RequiredColumnsConfig(
                    required_columns=["treatment", "outcome", "unit_id"]
                ),
                "binary_treatment": BinaryTreatmentConfig(
                    treatment_column="treatment"
                ),
            }
        )
    )
    
    # Run validation
    result = validator.validate(df)
    
    # Print results
    print(result.to_string(verbose=True))
    
    # Access specific information
    if not result.passed:
        print("\nValidation failed!")
        print(f"Failed checks: {', '.join(result.get_failed_checks())}")
        
        # Get all errors
        errors = [issue for issue in result.get_all_issues() 
                 if issue.severity.value == "error"]
        print(f"\nFound {len(errors)} errors:")
        for error in errors:
            print(f"  - {error.message}")


if __name__ == "__main__":
    # Example: Create sample data with some issues
    sample_data = pd.DataFrame({
        "unit_id": ["A", "A", "B", "B", "C", "C"],
        "time": [1, 2, 1, 2, 1, 2],
        "treatment": [0, 1, 0, 0, 0, 1],
        "outcome": [10.5, 12.3, 9.8, 10.1, None, 11.5],  # One missing value
        "covariate": [1, 1, 2, 2, 3, 3]
    })
    
    print("Validating sample data...")
    validate_dataframe_standalone(sample_data)
    
    # Example: Run the full validation graph
    print("\n" + "="*50 + "\n")
    print("Running validation graph...")
    graph = create_validation_graph()
    graph.fit(df=sample_data) 