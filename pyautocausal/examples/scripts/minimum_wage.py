import pandas as pd
from pathlib import Path
from pyautocausal.pipelines.example_graph import (
    causal_pipeline, _print_execution_summary, _export_outputs
)

# Load the minimum wage data
raw_data = pd.read_csv("pyautocausal/examples/data/mpdta.csv")

# Preprocess data by changing column names to match pipeline expectations
# The mpdta dataset has:
# - year: time variable
# - countyreal: unit identifier (county codes)  
# - lemp: log employment (outcome variable)
# - treat: treatment indicator (0/1) - but this is "absorbing" (always 1 once treated)
# - lpop: log population (covariate)
# - first.treat: first treatment year - this has the ACTUAL treatment timing
data = raw_data.rename(columns={
    "countyreal": "id_unit",  # county identifier
    "year": "t",              # time variable
    "lemp": "y"               # log employment as outcome
})

# Reconstruct the treatment variable properly using first.treat
# The original 'treat' column is absorbing (1 for all periods of units that ever get treated)
# We need to make it 0 before treatment and 1 from treatment onwards
print("Reconstructing treatment variable using first.treat column...")

# Create proper treatment indicator based on first.treat timing
def reconstruct_treatment(row):
    """Reconstruct treatment: 0 before first.treat, 1 from first.treat onwards"""
    if pd.isna(row['first.treat']) or row['first.treat'] == 0:
        # Never treated units
        return 0
    elif row['t'] >= row['first.treat']:
        # Treated in current period (treatment started)
        return 1
    else:
        # Not yet treated (before treatment start)
        return 0

data['treat'] = data.apply(reconstruct_treatment, axis=1)

# Keep additional covariates
data = data[["id_unit", "t", "treat", "y", "lpop"]]  

# Define output path
output_path = Path("pyautocausal/examples/outputs")
output_path.mkdir(parents=True, exist_ok=True)

# Initialize graph
graph = causal_pipeline(output_path)

# Save data for reference - same location as notebook
notebooks_path = output_path / "notebooks"
notebooks_path.mkdir(parents=True, exist_ok=True)
data_csv_path = notebooks_path / "minimum_wage.csv"
data.to_csv(data_csv_path, index=False)

print(f"\nProcessed data saved to {data_csv_path}")

# Run the causal pipeline
graph.fit(df=data)

# Results summary and export
print("\n======= Execution Summary =======")
_print_execution_summary(graph)
print("-" * 50)

_export_outputs(graph, output_path, datafile_name="minimum_wage.csv")
print("\n======= Minimum Wage Analysis Finished =======")
