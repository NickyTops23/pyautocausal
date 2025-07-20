import pandas as pd
from pathlib import Path
from pyautocausal.pipelines.example_graph import causal_pipeline

# Load Criteo data
df = pd.read_parquet("pyautocausal/examples/data/criteo_sample.parquet")

# Transform to PyAutoCausal conventions
df = df.rename(columns={
    'treatment': 'treat',      # Standard treatment column name
    'conversion': 'y'          # Standard outcome column name  
    # Alternative: use 'visit': 'y' for visit outcome
})

# Add unit identifier (required by PyAutoCausal)
df['id_unit'] = df.index

# Optional: Add time column if analyzing temporal patterns
# df['t'] = df['timestamp'].dt.year

# Features f0-f11 will be automatically detected as control variables
# No need to explicitly specify them - follows existing pattern

# Execute pipeline - will automatically route to uplift branch
output_path = Path('pyautocausal/examples/outputs/criteo_uplift')
output_path.mkdir(parents=True, exist_ok=True)

graph = causal_pipeline(output_path)
graph.fit(df=df)

print("Uplift modeling analysis complete!")
print(f"Results saved to: {output_path}")






