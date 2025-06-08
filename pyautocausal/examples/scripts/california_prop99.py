import pandas as pd
from pathlib import Path
from pyautocausal.pipelines.example_graph import (
    causal_pipeline, _print_execution_summary, _export_outputs
)


# The data already has the correct column names: id_unit, t, treat, y, plus control variables
data = pd.read_csv("pyautocausal/examples/data/california_prop99.csv")

data = data.rename(columns={"state": "id_unit", "year": "t", "treated": "treat", "cigsale": "y"})

data = data.fillna(-999)

# Define output path
output_path = Path("pyautocausal/examples/outputs")
output_path.mkdir(parents=True, exist_ok=True)

# Initialize graph
graph = causal_pipeline(output_path)

# Save data for reference - same location as notebook
notebooks_path = output_path / "notebooks"
notebooks_path.mkdir(parents=True, exist_ok=True)
data_csv_path = notebooks_path / "california_prop99.csv"
data.to_csv(data_csv_path, index=False)

print(f"Processed data saved to {data_csv_path}")

graph.fit(df=data)

# Results summary and export
print("\n======= Execution Summary =======")
_print_execution_summary(graph)
print("-" * 50)

_export_outputs(graph, output_path, datafile_name="california_prop99.csv")
print("\n======= Example Graph Run Finished =======")

