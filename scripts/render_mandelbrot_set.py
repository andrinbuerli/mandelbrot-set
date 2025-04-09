import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add argparse CLI options
cli = argparse.ArgumentParser(description="Render the Mandelbrot set and its derivative")
cli.add_argument("--width", type=int, default=1000, help="Width of the output image")
cli.add_argument("--height", type=int, default=1000, help="Height of the output image")
cli.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
cli.add_argument("--xmin", type=float, default=-2, help="Minimum x-value")
cli.add_argument("--xmax", type=float, default=1, help="Maximum x-value")
cli.add_argument("--ymin", type=float, default=-1.5, help="Minimum y-value")
cli.add_argument("--ymax", type=float, default=1.5, help="Maximum y-value")
cli.add_argument("--output", type=str, default="mandelbrot_plot.png", help="Output file name")
cli.add_argument("--zoomed", action="store_true", help="Zoom in on a smaller region")
cli.add_argument("--render_mag", action="store_true", help="Render magnitude of the final z value")
cli.add_argument("--render_derivative", action="store_true", help="Render log-scale of the derivative magnitude")
args = cli.parse_args()

# Define rendering parameters
width, height = args.width, args.height
xmin, xmax, ymin, ymax = args.xmin, args.xmax, args.ymin, args.ymax
if args.zoomed:
    xmin, xmax, ymin, ymax = -1.0, -0.5, -0.5, 0.0

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, render_mag=False):
    """Compute the Mandelbrot divergence iterations and its derivative."""
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    c = real[:, np.newaxis] + 1j * imag[np.newaxis, :]
    
    z = np.zeros_like(c, dtype=np.complex128)
    dz_dc = np.zeros_like(c, dtype=np.complex128)
    divergence_step = np.zeros(c.shape, dtype=int)
    
    for i in tqdm(range(max_iter), desc="Computing iterations"):
        dz_dc = 2 * z * dz_dc + 1
        z = z**2 + c
        mask = (np.abs(z) > 2) & (divergence_step == 0)
        divergence_step[mask] = i
    
    return  divergence_step if not render_mag else np.abs(z), np.abs(dz_dc)

# Compute Mandelbrot data
divergence_data, derivative_magnitude = compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, args.max_iter, args.render_mag)

# Select rendering mode
if args.render_mag:
    data_to_render = np.where(((~np.isnan(divergence_data)) & (divergence_data < 2.0)), divergence_data, 2.0)
    color_label = "Magnitude of final z value"
elif args.render_derivative:
    data_to_render = np.log1p(derivative_magnitude)
    color_label = "log(1 + |d(z)/d(c)|)"
else:
    data_to_render = divergence_data
    color_label = "Iterations to Diverge"

# Plot the result
plt.figure(figsize=(8, 8), dpi=300)
plt.imshow(
    data_to_render.T,
    extent=[xmin, xmax, ymin, ymax],
    cmap="hot",
    origin="lower",
    interpolation="none",
)
plt.colorbar(label=color_label)
plt.title(color_label + f" after {args.max_iter} Iterations")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.tight_layout()
plt.savefig(args.output)
