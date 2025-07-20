import pandas as pd
import pytest
import numpy as np

from pyautocausal.data_cleaning.autocleaner import AutoCleaner

@pytest.fixture
def sample_data_for_cleaning():
    """Creates a sample DataFrame with various issues to be cleaned."""
    np.random.seed(42)
    n_rows = 100
    data = pd.DataFrame({
        'treatment': np.random.choice([0, 1, 2], n_rows, p=[0.8, 0.15, 0.05]),  # Has an invalid value
        'outcome': np.random.normal(10, 2, n_rows),
        'age': np.random.randint(20, 60, n_rows),
        'city': np.random.choice(['A', 'B', 'C', None], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        're74': np.random.normal(5000, 1000, n_rows),
        're75': np.random.normal(5500, 1000, n_rows),
    })
    data.loc[0, 're74'] = np.nan  # Add a guaranteed NaN
    data = pd.concat([data, data.head(5)], ignore_index=True)  # Add duplicate rows
    data['city'] = data['city'].astype('object') # Ensure it's not categorical yet
    return data

def test_autocleaner_chaining_and_execution(sample_data_for_cleaning):
    """Tests the full pipeline of the AutoCleaner facade."""
    df = sample_data_for_cleaning.copy()
    initial_rows = len(df)
    initial_duplicates = df.duplicated().sum()

    # 1. Configure the AutoCleaner using the fluent API
    autocleaner = (
        AutoCleaner()
        .check_required_columns(required_columns=['treatment', 'outcome', 'age'])
        .check_column_types(expected_types={'age': 'int64', 'treatment': 'int64'})
        .check_for_missing_data(strategy='drop_rows', check_columns=['city', 're74'])
        .infer_and_convert_categoricals(ignore_columns=['treatment'])
        .drop_duplicates()
    )

    # 2. Execute the cleaning process
    cleaned_df, metadata = autocleaner.clean(df)

    # 3. Assertions
    # Check shape and dropped rows
    assert cleaned_df.shape[0] < initial_rows
    assert metadata.total_rows_dropped > 0

    # Check for dropped duplicates
    assert cleaned_df.duplicated().sum() == 0
    assert metadata.total_rows_dropped >= initial_duplicates

    # Check for dropped NA values
    assert cleaned_df['city'].isnull().sum() == 0
    assert cleaned_df['re74'].isnull().sum() == 0

    # Check for categorical conversion
    assert pd.api.types.is_categorical_dtype(cleaned_df['city'])

    # Check that original df is untouched
    assert df.shape[0] == initial_rows

    # Check metadata
    assert metadata.start_shape == df.shape
    assert metadata.end_shape == cleaned_df.shape
    assert len(metadata.transformations) > 0
    op_names = [t.operation_name for t in metadata.transformations]
    assert "drop_missing_rows" in op_names
    assert "drop_duplicate_rows" in op_names
    assert "convert_to_categorical" in op_names

def test_autocleaner_fill_missing_strategy(sample_data_for_cleaning):
    """Tests the 'fill' strategy for missing data."""
    df = sample_data_for_cleaning.copy()
    fill_value = -1

    autocleaner = (
        AutoCleaner()
        .check_for_missing_data(strategy='fill', check_columns=['city'], fill_value=fill_value)
    )
    cleaned_df, metadata = autocleaner.clean(df)

    assert cleaned_df['city'].isnull().sum() == 0
    assert (cleaned_df['city'] == fill_value).any()
    assert metadata.total_rows_dropped == 0
    op_names = [t.operation_name for t in metadata.transformations]
    assert "fill_missing_with_value" in op_names 