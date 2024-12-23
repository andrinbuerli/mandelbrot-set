import numpy as np
import plotly.graph_objects as go

# Define an exponential fitting function for curve fitting (optional, not used here)
def exponential(x, a, b):
    return a * np.exp(b * x)

# Parameters for searching near (-1.5, 0)
search_range = 0.01
x_min, x_max = -1.5 - search_range, -1.5 + search_range
y_min, y_max = -search_range, search_range
grid_resolution = 100
max_iter_search = 100

# Generate a grid of points near (-1.5, 0)
real_values = np.linspace(x_min, x_max, grid_resolution)
imag_values = np.linspace(y_min, y_max, grid_resolution)
grid_points = [(re, im) for re in real_values for im in imag_values]

# Find an unstable point
for re, im in grid_points:
    z = 0
    stable = True
    for _ in range(max_iter_search):
        z = z**2 + (re + im * 1j)
        if abs(z) > 2:
            stable = False
            break
    if not stable:
        unstable_point = (re, im)
        break

# Compute magnitudes for the unstable point over 100 iterations
max_iter_investigation = 10
unstable_point_magnitudes = [abs(0)]
z = 0
for _ in range(max_iter_investigation):
    z = z**2 + (unstable_point[0] + unstable_point[1] * 1j)
    unstable_point_magnitudes.append(abs(z))

# Create an interactive plot for the unstable point
x_data_unstable = np.arange(max_iter_investigation + 1)

fig = go.Figure()

# Add the sequence magnitudes
fig.add_trace(go.Scatter(
    x=x_data_unstable,
    y=unstable_point_magnitudes,
    mode='lines+markers',
    name=f"Point ({unstable_point[0]:.5f}, {unstable_point[1]:.5f})",
    line=dict(color='blue')
))

# Add the divergence threshold line
fig.add_trace(go.Scatter(
    x=x_data_unstable,
    y=[2] * len(x_data_unstable),
    mode='lines',
    name='Divergence Threshold',
    line=dict(color='red', dash='dash')
))

# Set plot titles and labels
fig.update_layout(
    title=f"Unstable Point Near (-1.5, 0) - 100 Iterations",
    xaxis_title="Iteration",
    yaxis_title="Magnitude",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly"
)

print(f"Unstable point: {unstable_point}")
# Show the interactive plot
fig.show()
