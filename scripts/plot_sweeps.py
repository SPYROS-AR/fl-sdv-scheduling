import pandas as pd
import plotly.express as px

# Load the ranked sweep data
df = pd.read_csv("final_sweep_rankings.csv")

# Create the parallel coordinates plot
fig = px.parallel_coordinates(
    df, 
    # Order the columns: Inputs first, then Outputs, ending with Utility
    dimensions=['alpha', 'beta', 'gamma', 'delta', 'network_mbs', 'drift', 'interference', 'final_loss', 'utility'],
    color='utility',
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Hyperparameter Sweep: Parallel Coordinates"
)

# Save as an interactive HTML file
fig.write_html("sweep_parallel_coordinates.html")
print("Saved interactive chart to sweep_parallel_coordinates.html")